#!/usr/bin/env python
from ...clip import CLIPVisionModelWithProjection
from ....utils import PREDEFINED_PATHS, load_embedding
from ....utils.train import get_thresholds, get_lora_merged_state_dict, get_codecnet_state_dict

from ...compressed.codecnet import CodecNet

# from kernl.model_optimization import optimize_model
import torch
from torch import nn
import reuse_partition

import triton
import triton.language as tl

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

torch.manual_seed(42)

first_pass_idx = torch.tensor([4, 7], dtype=torch.long, device='cuda')
second_pass_idx = torch.tensor([1, 2, 3, 5, 6, 8, 9], dtype=torch.long, device='cuda')

@triton.jit
def softmax_upto_exp_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, numerator, mask=col_offsets < n_cols)

@triton.jit
def softmax_from_exp_to_per_frame_kernel(output_ptr, input_ptr, mask_ptr, input_row_stride, mask_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    mask_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    mask_start_ptr = mask_ptr + mask_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=True)
    row[mask] = 0.0

    denominator = tl.sum(row, axis=0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, row / denominator, mask=col_offsets < n_cols)

def softmax_per_frame(x, masks):
    n_rows, n_cols = x.shape
    n_frames = masks.shape[0]
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_upto_exp_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    z = torch.empty((n_frames, n_rows, n_cols), dtype=torch.float32, device=x.device)
    softmax_from_exp_to_per_frame_kernel[(n_frames, n_rows)](
        z,
        y,
        masks,
        y.stride(0),
        masks.stride(0),
        z.stride(1),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return z



def gather_batched_states(gather_idx, lhs_states, rhs_states):
    '''
    args:
        gather_idx: [B, N]
        lhs_states: [B, N, dim]
        rhs_states: [1, N', dim], where N' is batched number of tokens
    returns:
        gathered_states: [B, N, dim]  
    '''
    _, _, dim = lhs_states.shape
    B, N = gather_idx.shape

    lhs_states = lhs_states.view(1, -1, dim)
    rhs_states = rhs_states.view(1, -1, dim)

    batched_states = torch.cat((lhs_states, rhs_states), dim=1).expand(B, -1, -1)

    gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, dim)
    batched_states = torch.gather(batched_states, dim=1, index=gather_idx)
    return batched_states


class OptimizedSimEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            config,
            original_encoder_layer,
            reuse_thresholds,
            is_first_layer=False,
            is_last_layer=False,
            use_compressed_info=False,
        ):
        super().__init__()

        self.config = config
        self.original_encoder_layer = original_encoder_layer

        self.reuse_thresholds = nn.Parameter(reuse_thresholds)

        token_count = (config.image_size // config.patch_size) ** 2 + 1

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        self.last_reference_states = None 
        self.last_reference_states_norm = None
        self.last_hidden_states = None
        self.last_query_states = None
        self.last_key_states = None
        self.last_value_states = None

        self.use_compressed_info = use_compressed_info
        if use_compressed_info:
            self.codecnet = CodecNet(
                input_shape=[3, 4, 16, 16],
                # Encoder
                e_spatial_channels=[(3, 16), (16, 32), (32, 64)],
                e_spatial_kernel_size=[1, 2, 2],
                e_temporal_channels_list=[[4, 4], [4, 4], [4, 4]],
                e_activation='relu',
                e_use_bn=True,
                # Decoder
                d_spatial_channels=[(64, 32), (64, 16), (32, 16)],
                d_spatial_kernel_size=[1, 2, 2],
            )


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
        hidden_states,
        hidden_states_norm,
        sorted_reuse_thresholds,
        output_maps=False,
    ):
        B, N, dim = hidden_states.shape

        reference_states_norm = self.last_reference_states_norm[:B].view(B, -1, dim)

        # We don't care for CLS from now on
        query_states = hidden_states[:, 1:]
        query_states_norm = hidden_states_norm[:, 1:]

        # [B, N-1, dim] @ [B, dim, N] => [B, N-1, N]
        similarity = torch.bmm(query_states_norm, reference_states_norm.transpose(1, 2))
        # Compute similarity score and find most similar idx
        # [B, N-1, N] => [B, N-1]
        most_similar_score, most_similar_idx = similarity.max(dim=-1)
        # Add offset to most similar idx considering reference tensor is flattened
        offset = torch.arange(B, dtype=torch.long, device=hidden_states.device) * N
        most_similar_idx += offset.unsqueeze(-1)

        reuse_map = most_similar_score >= sorted_reuse_thresholds # [B, N-1]

        reuse_cnts = torch.sum(reuse_map, dim=-1) # [B]
        reuse_bases = torch.cumsum(reuse_cnts, dim=0) # [B]
        reuse_bases = torch.cat((torch.tensor([0], dtype=torch.long, device=hidden_states.device), reuse_bases)) # [B+1]

        num_items = B * (N - 1)
        num_reuse = reuse_bases[-1].item()
        num_compute = num_items - num_reuse

        gather_idx = torch.empty((B, N-1), dtype=torch.long, device=hidden_states.device)
        partition_idx = torch.empty(num_items, dtype=torch.long, device=hidden_states.device)
        compute_idx = torch.empty(num_compute, dtype=torch.long, device=hidden_states.device)

        reuse_partition.forward(
            B,
            N-1,
            dim,
            num_items,
            num_compute,
            B*N + B, # compute offset [prev_tokens, cls_tokens, compute_toknens]
            reuse_map,
            reuse_cnts,
            reuse_bases,
            most_similar_idx,
            gather_idx,
            partition_idx,
        )

        compute_idx = partition_idx[num_reuse:]
        compute_idx = compute_idx.unsqueeze(-1).expand(-1, dim)

        compute_states = torch.gather(
            query_states.reshape(-1, dim),
            dim=0,
            index=compute_idx
        )
        compute_states_norm = torch.gather(
            query_states_norm.reshape(-1, dim),
            dim=0,
            index=compute_idx
        )

        # CLS tokens are always computed
        compute_states = torch.cat([hidden_states[:, 0], compute_states], dim=0)
        compute_states_norm = torch.cat([hidden_states_norm[:, 0], compute_states_norm], dim=0)
        cls_idx = torch.arange(B, dtype=torch.long, device=gather_idx.device) + B*N
        gather_idx = torch.cat([cls_idx.unsqueeze(-1), gather_idx], dim=-1)

        hidden_states = compute_states.unsqueeze(0) # [B*N, dim] => h', dim] [1, N', dim]
        hidden_states_norm = compute_states_norm.unsqueeze(0) # [B*Ng dim] => [N', dim] [1, N', dim]

        return reuse_map, gather_idx, hidden_states, hidden_states_norm

    def forward(
        self,
        hidden_states,
        *args,
        output_attentions=False,
        output_qkvs=False,
        output_maps=False,
        **kwargs,
    ):
        if not self.is_first_layer:
            # Gather index and compacted states received from previous layer
            gather_idxs, reference_states, reference_states_norm, hidden_states = hidden_states

        _, _, dim = hidden_states.shape

        # Run layer norm and QKV projection on (compacted) hidden states
        query_states, key_states, value_states, qkvs = self.layer_norm1_qkv_projection(
            hidden_states,
            output_qkvs=output_qkvs,
        )

        if not self.is_first_layer:
            # Repopulate hidden states and QKVs using gathered index
            if self.last_hidden_states is not None:
                hidden_states = gather_batched_states(gather_idxs, self.last_hidden_states, hidden_states)
                query_states = gather_batched_states(gather_idxs, self.last_query_states, query_states)
                key_states = gather_batched_states(gather_idxs, self.last_key_states, key_states)
                value_states = gather_batched_states(gather_idxs, self.last_value_states, value_states)

            self.last_hidden_states = hidden_states
            self.last_query_states = query_states
            self.last_key_states = key_states
            self.last_value_states = value_states

        B, N, dim = hidden_states.shape
        residual = hidden_states

        reuse_map = None
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            query_projected=query_states,
            key_projected=key_states,
            value_projected=value_states,
            output_attentions=True,
        )

        hidden_states = residual + attn_output

        # Final layer only needs to compute CLS token
        if self.is_last_layer:
            hidden_states = hidden_states[:, :1]
        else:
            # Reuse normalized hidden states
            hidden_states_norm = (hidden_states / hidden_states.norm(dim=-1, keepdim=True)) # [B, N, dim]

            if self.last_reference_states is None:
                # We have no cached states, so we need to compute all tokens
                reference_states = hidden_states
                reference_states_norm = hidden_states_norm
                reuse_map = torch.zeros((B, N), dtype=torch.bool, device=hidden_states.device)
                gather_idxs = torch.arange(B*N, dtype=torch.long, device=hidden_states.device).view(B, N)
            else:
                # Get rank of each token
                # Note the first frame does not need this
                # [B, H, N, N] => [B, H, N-1]
                cls_attn = attn_weights[:, :, 0, 1:]
                # [B, H, N-1] => [B, N-1]
                cls_attn = cls_attn.sum(dim=1)

                importance = torch.argsort(cls_attn, dim=-1, descending=False)
                ranks = torch.argsort(importance, dim=-1, descending=False)

                sorted_reuse_thresholds = torch.gather(
                    self.reuse_thresholds.unsqueeze(0).expand(B, -1),
                    dim=1,
                    index=ranks
                )

                if self.use_compressed_info:
                    compressed = kwargs['compressed']
                    compressed_map = self.codecnet(compressed)
                    compressed_map = compressed_map.view(B, -1)
                    sorted_compressed_map = torch.gather(
                        compressed_map,
                        dim=1,
                        index=ranks
                    )

                    sorted_reuse_thresholds = sorted_reuse_thresholds - sorted_compressed_map

                maps, gather_idxs, hidden_states, hidden_states_norm = self.stage_states(
                    hidden_states,
                    hidden_states_norm,
                    sorted_reuse_thresholds,
                    output_maps=output_maps,
                )

                reference_states = gather_batched_states(gather_idxs, self.last_reference_states, hidden_states)
                reference_states_norm = gather_batched_states(gather_idxs, self.last_reference_states_norm, hidden_states_norm)

            self.last_reference_states = reference_states
            self.last_reference_states_norm = reference_states_norm


        residual = hidden_states
        reference_states = hidden_states

        hidden_states = self.original_encoder_layer.layer_norm2(hidden_states)
        hidden_states = self.original_encoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if not self.is_last_layer:
            hidden_states = (gather_idxs, reference_states, reference_states_norm, hidden_states)

        return (hidden_states, attn_weights, qkvs, reuse_map)


def create_sim_model(
    base_model_name,
    reuse_model_name,
    checkpoint_path=None,
    use_lora=None,
    lowest_threshold=None,
    undo_sort=False,
    cache_dir=CACHE_DIR,
):
    model = CLIPVisionModelWithProjection.from_pretrained(base_model_name, cache_dir=cache_dir)

    use_sigmoid = PREDEFINED_PATHS['train'][reuse_model_name].get('use_sigmoid', False)
    use_softplus = PREDEFINED_PATHS['train'][reuse_model_name].get('use_softplus', False)
    use_fixed_reuse = PREDEFINED_PATHS['train'][reuse_model_name].get('use_fixed_reuse', False)
    use_compressed_info = PREDEFINED_PATHS['train'][reuse_model_name].get('use_compressed_info', False)

    thresholds_per_layer = get_thresholds(reuse_model_name, checkpoint_path, use_sigmoid=use_sigmoid, use_softplus=use_softplus)

    # Insert diffrate modules
    for layer_idx in range(len(model.vision_model.encoder.layers)):

        reuse_thresholds = thresholds_per_layer[layer_idx]

        original_encoder_layer = model.vision_model.encoder.layers[layer_idx]
        model.vision_model.encoder.layers[layer_idx] = OptimizedSimEncoderLayer(
            model.config,
            original_encoder_layer,
            reuse_thresholds,
            is_first_layer=layer_idx == 0,
            is_last_layer=layer_idx == len(model.vision_model.encoder.layers) - 1,
            use_compressed_info=use_compressed_info,
        )

    if use_lora is None:
        use_lora = PREDEFINED_PATHS['train'][reuse_model_name]['is_lora']

    if use_lora:
        state_dict = get_lora_merged_state_dict(reuse_model_name, checkpoint_path, cache_dir=cache_dir)

        ret = model.load_state_dict(state_dict, strict=False)

        missing_keys = ['threshold', 'codecnet']
        for k in ret.missing_keys:
            if not any(kw in k for kw in missing_keys):
                raise ValueError(f'Missing key {k}')

        expected_keywords = ['lora', 'sim_threshold', 'reuse_module', 'position_ids']
        for k in ret.unexpected_keys:
            if not any(kw in k for kw in expected_keywords):
                raise ValueError(f'Unexpected key {k}')

    if use_compressed_info:
        codecnet_state_dict = get_codecnet_state_dict(reuse_model_name)
        ret = model.load_state_dict(codecnet_state_dict, strict=False)
        assert len(ret.unexpected_keys) == 0

    return model


if __name__ == '__main__':
    BASE_MODEL_NAME='openai/clip-vit-large-patch14'
    CACHE_DIR = PREDEFINED_PATHS['root']['cache']
    REUSE_MODEL_NAME='how2qa/try46'
    # REUSE_MODEL_NAME='how2qa/try27'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_name', type=str, default=BASE_MODEL_NAME)
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR)

    device = 'cuda'

    reuse_model = create_sim_model(
        BASE_MODEL_NAME,
        REUSE_MODEL_NAME,
        cache_dir=CACHE_DIR)
    reuse_model = reuse_model.eval()
    reuse_model = reuse_model.to(device)
    # optimize_model(reuse_model)

    frames = load_embedding('/workspace/jupyter/frames.npz')

    frames = frames.to(device)

    NUM_BATCH = 16

    compressed = torch.randn((NUM_BATCH, 3, 4, 16, 16), device=device)

    with torch.no_grad():
        for frame in frames:
            _ = reuse_model(
                pixel_values=frame.unsqueeze(0).repeat(NUM_BATCH, 1, 1, 1),
                attention_mask=None,
                causal_attention_mask=None,
                compressed=compressed,
            )
