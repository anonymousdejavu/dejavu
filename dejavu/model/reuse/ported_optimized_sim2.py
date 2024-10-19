#!/usr/bin/env python
from transformers import CLIPVisionConfig
import numpy as np

from ..clip.modeling_optim_clip import CLIPVisionModelWithProjection
from ...utils.dataset import load_embedding
from ...utils import PREDEFINED_PATHS
from ...utils.train import get_thresholds, get_codecnet_state_dict

from ..compressed.codecnet_optim import codecnet

# from kernl.model_optimization import optimize_model
import torch
from torch import nn
import reuse_partition

CACHE_DIR = PREDEFINED_PATHS['root']['cache']
DEVICE = 'cuda'

# torch.manual_seed(43)

# first_pass_idx = torch.tensor([4, 7], dtype=torch.long, device='cuda')
# second_pass_idx = torch.tensor([1, 2, 3, 5, 6, 8, 9], dtype=torch.long, device='cuda')

def gather_stacked_batched_states(gather_idx, lhs_states, rhs_states, out=None):
    '''
    args:
        gather_idx: [B, N]
        lhs_states: [4, B, num_caches*N, dim], Cache
        rhs_states: [4, 1, N', dim], N' = B(cls_tokens) + (reference indexes)
    returns:
        gathered_states: [B, N, dim]  
    '''
    _, B, Nl, dim = lhs_states.shape
    _, _, Nr, _ = rhs_states.shape
    # B, N = gather_idx.shape
    BNl = B*Nl

    batched_states = torch.empty(4,1,BNl+Nr,dim, device=gather_idx.device)

    batched_states_tmp = batched_states[:,:,:BNl].view(4,B,Nl,dim)
    batched_states_tmp[:] = lhs_states
    batched_states[:,:,BNl:] = rhs_states
    batched_states = batched_states.expand(-1,B,-1,-1)  # 4,B,N+,dim

    gather_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(4, -1, -1, dim) # 4,B,N,dim
    batched_states = torch.gather(batched_states, dim=-2, index=gather_idx, out=out)
    return batched_states


def gather_batched_states(gather_idx, lhs_states, rhs_states, out=None):
    '''
    args:
        gather_idx: [B, N]
        lhs_states: [B, num_caches*N, dim], Cache
        rhs_states: [1, N', dim], N' = B(cls_tokens) + (reference indexes)
    returns:
        gathered_states: [B, N, dim]  
    '''
    B, Nl, dim = lhs_states.shape
    _, Nr, _ = rhs_states.shape
    # B, _ = gather_idx.shape

    BNl = B*Nl

    batched_states = torch.empty(1,BNl+Nr,dim, device=gather_idx.device)

    batched_states_tmp = batched_states[:,:BNl].view(1,B,Nl,dim)
    batched_states_tmp[:] = lhs_states
    batched_states[:,BNl:] = rhs_states
    batched_states = batched_states.expand(B,-1,-1)

    # lhs_states = lhs_states.view(1, -1, dim)
    # rhs_states = rhs_states.view(1, -1, dim)
    # batched_states = torch.cat((lhs_states, rhs_states), dim=-2).expand(B, -1, -1)  # 4,1->B,B+,dim

    gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, dim)
    batched_states = torch.gather(batched_states, dim=-2, index=gather_idx, out=out)
    return batched_states


class OptimizedSimEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            config,
            original_encoder_layer,
            reuse_thresholds,
            # TODO: get reuse map input. reuse map input
            reuse_map=None,
            is_first_layer=False,
            is_last_layer=False,
            use_compressed_info=False,
            # is_first_frame=True,
            # max_batch_size=10,
        ):
        super().__init__()

        self.eps = torch.tensor(1e-6, device='cuda')
        # self.dbg_is_first_inference = True
        # self.reuse_map = reuse_map

        self.config = config
        self.original_encoder_layer = original_encoder_layer

        self.reuse_thresholds = nn.Parameter(reuse_thresholds)

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.token_num_per_side = config.image_size // config.patch_size
        self.token_count = self.token_num_per_side ** 2 + 1

        self.use_compressed_info = use_compressed_info
        if use_compressed_info:
            self.codecnet = codecnet(
                input_shape=[6, 4, self.token_num_per_side, self.token_num_per_side],
                # Encoder
                e_spatial_channels=[(6, 16), (16, 32), (32, 64)],
                e_spatial_kernel_size=[1, 2, 2],
                e_temporal_channels_list=[[4, 4], [4, 4], [4, 4]],
                e_activation='relu',
                e_use_bn=True,
                e_patch_per_side=self.token_num_per_side,
                # Decoder
                d_spatial_channels=[(64, 32), (64, 16), (32, 16)],
                d_spatial_kernel_size=[1, 2, 2],
            )

        # self.cache_access_pattern = [
        #     torch.tensor([0], dtype=torch.long),
        #     torch.tensor([0,1], dtype=torch.long),
        #     torch.tensor([0,2], dtype=torch.long),
        #     torch.tensor([1,2], dtype=torch.long),
        #     ]

    
    def get_device(self):
        return self.reuse_thresholds.device

    # LayerNorm + Projection
    def layer_norm1_qkv_projection(
        self,
        hidden_states,
        output_qkvs=False
    ):
        hidden_states = self.original_encoder_layer.layer_norm1(hidden_states)

        # Projection
        self_attn = self.original_encoder_layer.self_attn

        bsz, tgt_len, embed_dim = hidden_states.size()
        query_projected = self_attn.q_proj(hidden_states)
        key_projected = self_attn.k_proj(hidden_states)
        value_projected = self_attn.v_proj(hidden_states)

        if output_qkvs:
            qkvs = (
                query_projected.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim),
                key_projected.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim),
                value_projected.view(bsz, tgt_len, self_attn.num_heads, self_attn.head_dim)
            )
        else:
            qkvs = None

        return query_projected, key_projected, value_projected, qkvs

    # Self Attention
    def self_attn(
        self,
        hidden_states,
        query_projected,
        key_projected,
        value_projected,
        output_attentions=False,
    ):
        self_attn = self.original_encoder_layer.self_attn
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = query_projected * self_attn.scale
        key_states = self_attn._shape(key_projected, -1, bsz)
        value_states = self_attn._shape(value_projected, -1, bsz)

        proj_shape = (bsz * self_attn.num_heads, -1, self_attn.head_dim)
        query_states = self_attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self_attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self_attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self_attn.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self_attn.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self_attn.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self_attn.num_heads, tgt_len, self_attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self_attn.num_heads, tgt_len, self_attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self_attn.num_heads, tgt_len, self_attn.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self_attn.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def stage_states(
        self,
        list_hidden_states,           # 4,B,N,dim
        list_hidden_states_norm,      # 4,B,N,dim
        reference_state_caches,       # B,3,N,dim
        list_sorted_reuse_thresholds, # 4,B,N
        is_first_four,
        output_maps=False,
    ):
        _, B, N, dim = list_hidden_states.shape

        list_gather_idxs = torch.empty((4,B,N), dtype=torch.long, device=list_hidden_states.device)
        # cache, cls, compute idx
        # this is for cls tokens
        list_gather_idxs[:,:,0] = torch.arange(B, dtype=torch.long, device=list_hidden_states.device)
        list_gather_idxs[:1,:,0] += B*N
        list_gather_idxs[1:4,:,0] += 2*B*N

        list_reuse_map = torch.empty((4,B,N-1), dtype=torch.bool, device=list_hidden_states.device)

        reuse_bases = torch.zeros(B+1, dtype=torch.long, device=list_hidden_states.device) # B+1

        ret_list_hidden_states = []

        # 0,4,2:
        # - 0 -> 0
        # - 0,4,4 -> 0:1
        # - 0,2,4 -> 0:1
        # - 0,2,4 -> 1:2

        # XXX: if video is ordered to 4,2,1,3, range need to be fixed.
        for i in range(4):
            num_caches = 1 + (i!=0)
            hidden_states = list_hidden_states[i]           # B,N,dim
            hidden_states_norm = list_hidden_states_norm[i] # B,N,dim
            query_states = hidden_states[:,1:]              # B,N-1,dim
            query_states_norm = hidden_states_norm[:,1:]    # B,N-1,dim
            # B,T,N,dim -> B,{1,2},N,dim -> B,{1,2}*N,dim
            start = (0^is_first_four)*(i==3) + (1^is_first_four)*((2-i) + (i>=2))
            reference_state_cache = reference_state_caches[:,start:start+num_caches] \
                    .view(B, num_caches*self.token_count, self.config.hidden_size)

            similarity = torch.bmm(query_states_norm, reference_state_cache.transpose(-1, -2)) # B,N-1,num_caches*N

            # most_similar_idx -> partition_idx[:reuse]
            most_similar_score, most_similar_idx = similarity.max(dim=-1) # B,N-1 / value < num_caches*N

            offset = torch.arange(B, dtype=torch.long, device=hidden_states.device) * num_caches * N
            most_similar_idx += offset.unsqueeze(-1)

            list_reuse_map[i] = most_similar_score >= list_sorted_reuse_thresholds[i] # [B, N-1]

            # TODO: update list_reuse_map here
            # list_reuse_map[i] = self.reuse_map[i].expand(B,N-1)

            reuse_cnts = torch.sum(list_reuse_map[i], dim=-1) # [B]

            reuse_bases[1:] = torch.cumsum(reuse_cnts, dim=0)

            num_items = B * (N - 1)
            num_reuse = reuse_bases[-1].item()
            num_compute = num_items - num_reuse

            gather_idx = torch.empty((B, N-1), dtype=torch.long, device=hidden_states.device)
            partition_idx = torch.empty(num_items, dtype=torch.long, device=hidden_states.device)

            reuse_partition.forward(
                B,
                N-1,
                dim,
                num_items,
                num_compute,
                num_caches*B*N + B, # compute offset [prev_tokens, cls_tokens, compute_toknens]
                list_reuse_map[i],
                reuse_cnts,
                reuse_bases,
                most_similar_idx,
                gather_idx,     # B,N-1
                partition_idx,  # B*N-1
            )

            # gather_idx = most_similar_idx

            compute_idx = partition_idx[num_reuse:]
            compute_idx = compute_idx.unsqueeze(-1).expand(-1, dim)

            # TODO: copies
            compute_states = torch.gather(
                query_states.reshape(-1, dim),
                dim=0,
                index=compute_idx
            ) # N'',dim
            compute_states_norm = torch.gather(
                query_states_norm.reshape(-1, dim),
                dim=0,
                index=compute_idx
            )

            # CLS tokens are always computed
            # hidden_states : B,N,dim
            # B,dim + N'',dim -> N',dim
            compute_states = torch.cat([hidden_states[:, 0], compute_states], dim=0) # N',dim
            compute_states_norm = torch.cat([hidden_states_norm[:, 0], compute_states_norm], dim=0)
            list_gather_idxs[i,:,1:] = most_similar_idx

            hidden_states = compute_states.unsqueeze(0)           # 1,N',dim
            hidden_states_norm = compute_states_norm.unsqueeze(0) # 1,N',dim

            ret_list_hidden_states.append(hidden_states)

            if i == 0:
                idx = 2*is_first_four # is_first_four: update 1,2 else: updqte 0,1
                reference_state_caches[:,idx] = gather_batched_states(
                    list_gather_idxs[i], 
                    reference_state_cache, 
                    hidden_states_norm, 
                    out=reference_state_caches[:,1])
            elif i == 1: # update 1
                gather_batched_states(
                    list_gather_idxs[i], 
                    reference_state_cache, 
                    hidden_states_norm, 
                    out=reference_state_caches[:,1])

        return list_reuse_map, list_gather_idxs, ret_list_hidden_states
   
    # Currently it assumes 4 frame inputs, ((sum(Nij') for i in 4) for j in B),P,dim
    # Cache is (B,3,N,dim) shape
    # gather_idx is (4,B,N)
    def forward(
        self,
        list_hidden_states, # 4*B,N,dim tensors   for the first layer. this is due to preproc
                            # (4,B,N gather_idxs, len4[1,N',dim tensors])   for the rest
        *args,
        output_attentions=False,
        output_qkvs=False,
        output_maps=False,
        **kwargs, # kwargs["cache"] as caches. h,q,k,v,ref_norm 
                  # 5,B,3,N,dim
    ):
        if not self.is_first_layer:
            # Gather index and compacted states received from previous layer
            # reference_states is a tensor after attention layer of previous layer
            list_gather_idxs, list_hidden_states, intervals = list_hidden_states
        # else:
            # XXX: not needed if it is batched
            # B4,N,dim = list_hidden_states.shape
            # list_hidden_states = list_hidden_states.view(4,B4//4,N,dim)

        list_attn_weights = []
        list_hidden_states_buf = []

        cache = kwargs["cache"] # 5,B,3,N,dim
        is_first_four = kwargs["is_first_four"]
        _, B, _, N, dim = cache.shape
        # cache_shape = (B, 3, N, dim) # B,3(buf_size),N,dim

        # hqkv_caches = torch.empty((4,*cache_shape), device=cache.device) # 4,B,3,N,dim
        # reference_state_caches = torch.empty(cache_shape, device=cache.device) # B,3,N,dim
        
        hqkv_caches = cache[:4]
        reference_state_caches = cache[4]

        # print(list_hidden_states.shape)
        # TODO: qkv here, and split to 4 hqkvs
        list_query_states, list_key_states, list_value_states, qkvs = self.layer_norm1_qkv_projection(
            list_hidden_states,
            output_qkvs=output_qkvs,
        ) # 1,sigN',dim or 4,B,N,dim

        # qkvh,1,T*N  ,dim    qkvh,T*B,N,dim
        # 4   ,1,sigN',dim or 4   ,4*B,N,dim
        list_hqkv_states = torch.stack([list_hidden_states, list_query_states, list_key_states, list_value_states])

        if not self.is_first_layer:
            # [hqkv,1,N,dim]
            list_hqkv_states = list_hqkv_states.split(intervals, dim=-2)
        else:
            # [hqkv,B,N,dim]
            list_hqkv_states = list_hqkv_states.split(B,dim=1)

        # list_hidden_states = list_hidden_states.split(intervals, dim=-2)

        # XXX: if video is not ordered to 3,1,0,2, range need to be fixed.
        for i in range(4):
            # hqkv_states = list_hqkv_states[:,i]

            # hidden_states = list_hidden_states[i] # B,N,dim or 1,N',dim
            # # Q,K,V or Q',K',V'
            # query_states, key_states, value_states, qkvs = self.layer_norm1_qkv_projection(
            #     hidden_states,
            #     output_qkvs=output_qkvs,
            # )

            hqkv_states = list_hqkv_states[i]
            # restore QKV
            # if not self.is_first_layer:
            if not self.is_first_layer:
                num_caches = 1 + (i!=0)
                # 4,B,3,N,dim -> 4,B,num_caches,N,dim -> 4,B,num_caches*N,dim
                # 0:1 / 0:2 / 0:2 / 1:3
                # 2:3 / 1:3 / 1:3 / 0:2
                # is_first_four is True or False
                start = (is_first_four^0)*(i==3) + (is_first_four^1)*((2-i) + (i>=2))
                hqkv_cache = hqkv_caches[:,:,start:start+num_caches] \
                            .view(4,B,num_caches*self.token_count,self.config.hidden_size) 

                # hqkv_states = torch.stack((hidden_states,query_states,key_states,value_states)) # 4,1,N',dim
                if i == 0:
                    idx = 2*is_first_four # 1-(-1)**is_first_four
                    hqkv_states = hqkv_caches[:,:,idx] = gather_stacked_batched_states(list_gather_idxs[i],
                                                  hqkv_cache,
                                                  hqkv_states,
                                                  out=hqkv_caches[:,:,1])
                elif i == 1:
                    hqkv_states = gather_stacked_batched_states(list_gather_idxs[i],
                                                  hqkv_cache,
                                                  hqkv_states,
                                                  out=hqkv_caches[:,:,1])
                else:
                    hqkv_states = gather_stacked_batched_states(list_gather_idxs[i],
                                                                hqkv_cache,
                                                                hqkv_states)

            # TODO: pre qkv, so split << 
            # B,N,dim
            # print(f'hqkv_states, {hqkv_states.shape}')
            hidden_states = hqkv_states[0]
            query_states = hqkv_states[1]
            key_states = hqkv_states[2]
            value_states = hqkv_states[3]
   
            # B,N,dim
            residual = hidden_states

            attn_output, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                query_projected=query_states,
                key_projected=key_states,
                value_projected=value_states,
                output_attentions=True,
            )

            hidden_states = residual + attn_output

            list_attn_weights.append(attn_weights)
            list_hidden_states_buf.append(hidden_states)
        
        # TODO: malloc
        list_hidden_states = torch.stack(list_hidden_states_buf) # 4,B,N,dim
        list_attn_weights = torch.stack(list_attn_weights)   # 4,B,nh,N,N

        _,B,N,dim = list_hidden_states.shape

        # --- list_attn_weights, list_hidden_states are used below
        
        list_maps = None

        # Final layer only needs to compute CLS token
        if self.is_last_layer:
            list_hidden_states = list_hidden_states[:, :, :1] # 4,B,1,dim
        else:
            # Reuse normalized hidden states
            list_hidden_states_norm = list_hidden_states / (list_hidden_states.norm(dim=-1, keepdim=True) + self.eps)

            # Get rank of each token
            # Note the first frame does not need this
            # [4, B, H, N, N] => [4, B, H, N-1]
            list_cls_attn = list_attn_weights[:, :, :, 0, 1:]
            # [4, B, H, N-1] => [4, B, N-1]
            list_cls_attn = list_cls_attn.sum(dim=-2)

            list_importance = torch.argsort(list_cls_attn, dim=-1, descending=True) # 4,B,N-1
            list_ranks = torch.argsort(list_importance, dim=-1, descending=False)   # 4,B,N-1


            reuse_threshold = self.reuse_thresholds.unsqueeze(0).unsqueeze(0).expand(4, B, -1)
            # if self.dbg_is_first_inference:
            #     # reuse_threshold = self.reuse_thresholds.unsqueeze(0).unsqueeze(0).repeat(4, B, 1)
            #     # reuse_threshold[0] = torch.full((B,N-1), 10, device=list_ranks.device)
            #     reuse_threshold = torch.full((4,B,N-1), 10, device=list_ranks.device)
            #     self.dbg_is_first_inference = False

            sorted_reuse_thresholds = torch.gather(
                reuse_threshold,
                dim=-1,
                index=list_ranks
            )

            if self.use_compressed_info:
                compressed = kwargs['compressed'] # 4,B,3+3,T,H,W
                # assume compressed data is concatenated with one_hot
                compressed = compressed.view(4*B,6,4,self.token_num_per_side, self.token_num_per_side)
                compressed_map = self.codecnet(compressed)
                compressed_map = compressed_map.view(4,B,-1)
                # compressed_map = compressed_map.view(B, -1)
                scaled_compressed_map = (compressed_map - 0.5) * 2
                sorted_reuse_thresholds = sorted_reuse_thresholds - scaled_compressed_map

            
            # compress hidden_states -> only for computation
            # gather_idxs is a recover code
            # ! output list_hiden_states is not a tensor, but a list of tensors with different 0th dimension
            list_maps, list_gather_idxs, list_hidden_states = self.stage_states(
                list_hidden_states,      # 4,B,N,dim
                list_hidden_states_norm, # 4,B,N,dim
                reference_state_caches,
                sorted_reuse_thresholds, # 4,B,N
                output_maps=output_maps,
                is_first_four=is_first_four,
            )

            # reference_states = gather_batched_states(gather_idxs, self.last_reference_states, hidden_states)
            # reference_states_norm = gather_batched_states(gather_idxs, self.last_reference_states_norm, hidden_states_norm)

            # self.update_cache(self.last_reference_states, reference_states, i)
            # self.update_cache(self.last_reference_states_norm, reference_states_norm, i)

            # [1,N',dim]
            intervals = []
            for hidden_states in list_hidden_states:
                intervals.append(hidden_states.size(-2))
            list_hidden_states = torch.concat(list_hidden_states, dim=-2)

        residual = list_hidden_states
        list_hidden_states = self.original_encoder_layer.layer_norm2(list_hidden_states)
        list_hidden_states = self.original_encoder_layer.mlp(list_hidden_states)
        list_hidden_states = residual + list_hidden_states

        # for i in range(4):
        #     hidden_states = list_hidden_states[i]
            
        #     residual = hidden_states
        #     hidden_states = self.original_encoder_layer.layer_norm2(hidden_states)
        #     hidden_states = self.original_encoder_layer.mlp(hidden_states)
        #     hidden_states = residual + hidden_states

        #     list_hidden_states[i] = hidden_states

        if not self.is_last_layer:
            # list_hidden_states = list_hidden_states.split(intervals, dim=-2)
            list_hidden_states = (list_gather_idxs, list_hidden_states, intervals)

        return (list_hidden_states, attn_weights, qkvs, list_maps)

def create_opt_sim_model(
    reuse_model_name,
    cache_dir,
    device='cuda',
    dbg=False,
):
    # -- config
    base_model_name = PREDEFINED_PATHS['train'][reuse_model_name]["base_model_name"]
    use_compressed_info = PREDEFINED_PATHS['train'][reuse_model_name].get("use_compressed_info", False)

    use_sigmoid = PREDEFINED_PATHS['train'][reuse_model_name].get('use_sigmoid', False)
    use_softplus = PREDEFINED_PATHS['train'][reuse_model_name].get('use_softplus', False)
    use_fixed_reuse = PREDEFINED_PATHS['train'][reuse_model_name].get('use_fixed_reuse', False)

    # -- initialize model
    model = CLIPVisionModelWithProjection.from_pretrained(base_model_name, cache_dir=cache_dir)

    thresholds_per_layer = get_thresholds(reuse_model_name, use_sigmoid=use_sigmoid, use_softplus=use_softplus)

    def load_reuse_map(reuse_model_name):
        reuse_rate_path = f"/workspace/jupyter/tmp/{reuse_model_name.replace('/', '_')}.npy"
        reuse = np.load(reuse_rate_path)
        reuse = torch.tensor(reuse, device=device)
        return reuse
    # reuse_map = load_reuse_map(reuse_model_name)
    # reuse_map = torch.zeros((12,4,196))

    for layer_idx in range(len(model.vision_model.encoder.layers)):

        reuse_thresholds = thresholds_per_layer[layer_idx]
        #DBG
        if dbg:
            reuse_thresholds = torch.full((196,), 10.0)

        original_encoder_layer = model.vision_model.encoder.layers[layer_idx]
        model.vision_model.encoder.layers[layer_idx] = OptimizedSimEncoderLayer(
            model.config,
            original_encoder_layer,
            reuse_thresholds,
            # reuse_map[layer_idx],
            is_first_layer=layer_idx == 0,
            is_last_layer=layer_idx == len(model.vision_model.encoder.layers) - 1,
            use_compressed_info=use_compressed_info,
        )

    # model key
    #       base_model.model.vision_model.encoder.layers.0.codecnet.preprocessing.bn.weight
    # loaded codecnet key
    # model.base_model.model.vision_model.encoder.layers.0.codecnet.preprocessing.bn.weight
    if use_compressed_info:
        codecnet_state_dict = get_codecnet_state_dict(reuse_model_name, is_optim=True)
        ret = model.load_state_dict(codecnet_state_dict, strict=False)

        dbg = 0
        for k in ret.unexpected_keys:
            dbg += 1
            if dbg == 10: 
                break

        assert len(ret.unexpected_keys) == 0

    return model
