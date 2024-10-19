import torch
from torch import nn
from ..model.clip import CLIPVisionModelWithProjection
from ..utils.train import get_monotonic_threshold
from peft import LoraConfig, get_peft_model
from ..model.compressed.codecnet import CodecNet

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

torch.manual_seed(42)

class ReuseThreshold(nn.Module):
    def __init__(self, 
                 kept_num, 
                 s=40, 
                 use_sigmoid=False, 
                 use_softplus=False, 
                 use_fixed_reuse=False, 
                 use_compressed_info=False):
        super().__init__()
        # self.sim_threshold = nn.Parameter(torch.zeros((kept_num,)), requires_grad=True)
        self.kept_num = kept_num
        self.use_sigmoid = use_sigmoid
        self.use_softplus = use_softplus
        self.use_fixed_reuse = use_fixed_reuse
        self.use_compressed_info = use_compressed_info
        
        self.s = s # s was 10 for LEOPARD tanh, but since tanh is more steeper
        
        if not use_fixed_reuse:
            self.sim_threshold = nn.Parameter(torch.Tensor(kept_num))
            # nn.init.uniform_(self.sim_threshold, a=0.0, b=0.3 / kept_num)
            nn.init.xavier_normal_(self.sim_threshold)
        else:
            self.reuse_map = nn.Parameter(torch.Tensor(kept_num))
            # nn.init.uniform_(self.reuse_map, a=0, b=1.0/kept_num)

    def forward(self, x, x_reuse, sim, compressed_map=None):
        if self.use_fixed_reuse:
            reuse_map = get_monotonic_threshold(self.reuse_map, use_sigmoid=self.use_sigmoid, use_softplus=self.use_softplus, use_fixed_reuse=True)
            # monotonically decreasing reuse_map 
            reuse_map = torch.stack([reuse_map for _ in range(x.shape[0])])
            reuse_map = torch.sigmoid(self.s * reuse_map)
        
        else:
            threshold = get_monotonic_threshold(self.sim_threshold, use_sigmoid=self.use_sigmoid, use_softplus=self.use_softplus)
            # When similarity is lower than threshold, use newly computed value
            reuse_decision = sim - threshold
            if self.use_compressed_info:
                scaled_compressed_map = (compressed_map - 0.5) * 2
                reuse_decision += scaled_compressed_map
            
            reuse_map = torch.sigmoid(self.s * reuse_decision)

        reuse_map_unsqueezed = reuse_map.unsqueeze(-1)
        # (N, T) => (N, T, 1) * (N, T, dim) => (N, T, dim)
        term_reuse = reuse_map_unsqueezed * x_reuse
        # When similarity is lower than threshold, reuse
        term_x = (1 - reuse_map_unsqueezed) * x
        output = term_x + term_reuse

        return reuse_map, output


class SimTable(nn.Module):
    def __init__(self, 
                 kept_num, 
                 use_sigmoid=False, 
                 use_softplus=False, 
                 use_fixed_reuse=False, 
                 use_compressed_info=False):
        super().__init__()
        self.cached_input = None
        self.cached_value = None
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.threshold = ReuseThreshold(kept_num, 
                                        use_sigmoid=use_sigmoid, 
                                        use_softplus=use_softplus, 
                                        use_fixed_reuse=use_fixed_reuse,
                                        use_compressed_info=use_compressed_info)
        self.use_compressed_info = use_compressed_info

    def query(self, query_input, query_value, compressed_map=None):
        B, N, dim = query_input.shape
        assert compressed_map is None or compressed_map.shape == (B, N)

        # Find cosine similarity between query and cached input
        normalized_cached_input = self.cached_input / self.cached_input.norm(dim=-1, keepdim=True)
        normalized_query_input = query_input / query_input.norm(dim=-1, keepdim=True)

        similarity = normalized_query_input @ normalized_cached_input.transpose(-1, -2)

        # [B, unimportant, important] => [B, unimportant, 1]
        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        similar_input = torch.gather(
            self.cached_input,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)
        reuse_map, reuse_input = self.threshold(
            x=query_input,
            x_reuse=similar_input,
            sim=most_similar_score,
            compressed_map=compressed_map
        )

        similar_value = torch.gather(
            self.cached_value,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)
        reuse_map, reuse_value = self.threshold(
            x=query_value,
            x_reuse=similar_value,
            sim=most_similar_score,
            compressed_map=compressed_map
        )

        return reuse_map, reuse_input, reuse_value

    def insert(self, cached_input, cached_value):
        self.cached_input = cached_input
        self.cached_value = cached_value

class SimReuseModule(nn.Module):
    def __init__(
            self,
            original_layer_norm2,
            original_mlp,
            kept_num,
            exclude_cls=True,
            use_sigmoid=False,
            use_softplus=False,
            use_fixed_reuse=False,
            use_compressed_info=False,
        ):
        super().__init__()
        self.layer_norm2 = original_layer_norm2
        self.mlp = original_mlp
        self.kept_num = kept_num
        self.exclude_cls = exclude_cls
        if exclude_cls:
            kept_num -= 1
        self.table = SimTable(kept_num, 
                              use_sigmoid=use_sigmoid, 
                              use_softplus=use_softplus, 
                              use_fixed_reuse=use_fixed_reuse, 
                              use_compressed_info=use_compressed_info)

    def forward(
            self,
            hidden_states,
            is_first_frame=False,
            compressed_map=None,
            **kwargs
        ):
        B, N, _ = hidden_states.shape

        # Before MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # After MLP
        hidden_states = residual + hidden_states

        if not is_first_frame:
            # Check which tokens should be reused into which value
            if self.exclude_cls:
                cls_residual = residual[:, 0:1]
                residual = residual[:, 1:]
                cls_state = hidden_states[:, 0:1]
                hidden_states = hidden_states[:, 1:]

            reuse_map, residual, reused_states = self.table.query(
                query_input=residual,
                query_value=hidden_states,
                compressed_map=compressed_map
            )

            if self.exclude_cls:
                residual = torch.cat((cls_residual, residual), dim=1)
                hidden_states = torch.cat((cls_state, reused_states), dim=1)
                reuse_map = torch.cat(
                    (
                        torch.zeros((B, 1), dtype=torch.bool, device=reuse_map.device),
                        reuse_map
                    ),
                    dim=1
                )
            self.last_reuse_map = reuse_map

        self.table.insert(residual, hidden_states)

        return hidden_states

    def clear(self):
        self.table.clear()

class SimEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            original_encoder_layer,
            prune_kept_num,
            merge_kept_num,
            kept_num, # min(prev, prune, merge)
            undo_sort=False,
            use_sigmoid=False,
            use_softplus=False,
            use_fixed_reuse=False,
            use_compressed_info=False,
        ):
        super().__init__()
        self.original_encoder_layer = original_encoder_layer
        self.prune_kept_num = prune_kept_num
        self.merge_kept_num = merge_kept_num
        self.reuse_module = SimReuseModule(
            original_encoder_layer.layer_norm2,
            original_encoder_layer.mlp,
            kept_num,
            use_sigmoid=use_sigmoid,
            use_softplus=use_softplus,
            use_fixed_reuse=use_fixed_reuse,
            use_compressed_info=use_compressed_info
        )
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

        # Used for debuggging for now
        self.undo_sort = undo_sort
        self.use_compressed_info = use_compressed_info  # for debugging

    def forward(
            self,
            hidden_states,
            *args,
            output_attentions=False,
            output_qkvs=False,
            output_maps=False,
            **kwargs,
        ):
        B, N, dim = hidden_states.shape

        # Same as original encoder upto self attention
        residual = hidden_states

        hidden_states = self.original_encoder_layer.layer_norm1(hidden_states)
        hidden_states, attn_weights, qkvs = self.original_encoder_layer.self_attn(
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=True,
            output_qkvs=output_qkvs,
        )
        hidden_states = residual + hidden_states # [B, N, dim]

        maps = None

        # Size of attn_weights is [B, H, N, N]
        cls_attn = attn_weights[:, :, 0, 1:] # [B, H, N-1]
        cls_attn = cls_attn.mean(dim=1)      # [B, N-1]
        _, attn_idx = torch.sort(cls_attn, descending=True)
        cls_idx = torch.zeros((B, 1), device=attn_idx.device, dtype=torch.long)
        # assert cls_idx.device == attn_idx.device
        idx = torch.cat((cls_idx, attn_idx + 1), dim=1) # [B, N]

        if output_maps:
            # Maps are mapping from [B, N] to pruned and sorted output
            maps = idx

        # Sort by attention weights
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            # Shape: [B, N] => [B, N, 1] => [B, N, dim]
            index=idx.unsqueeze(-1).expand(-1, -1, dim),
        )

        # [B, N, dim] => [B, prune_kept_num, dim]
        if self.prune_kept_num < N:
            hidden_states = self.prune(hidden_states)

        # [B, prune_kept_num, dim] => [B, merge_kept_num, dim]
        if self.merge_kept_num < hidden_states.shape[1]:
            hidden_states = self.merge(hidden_states)

        if self.use_compressed_info:
            compressed_input = kwargs['compressed_input']
            compressed_map = self.codecnet(compressed_input)
            compressed_map = compressed_map.view(B, -1)
        
            sorted_compressed_map = torch.gather(
                compressed_map, # 196
                dim=1,
                index=attn_idx
            )
        else:
            sorted_compressed_map = None

        hidden_states = self.reuse_module(
            hidden_states,
            compressed_map=sorted_compressed_map,
            **kwargs,
        )

        # Added : undo_sort
        _, reversal_idx = torch.sort(idx, dim=1)
        hidden_states = torch.gather(
            hidden_states,
            dim=1,
            # Shape: [B, N] => [B, N, 1] => [B, N, dim]
            index=reversal_idx.unsqueeze(-1).expand(-1, -1, dim),
        )

        return (hidden_states, attn_weights, qkvs, maps)

    def prune(self, hidden_states):
        return hidden_states[:, :self.prune_kept_num]

    def merge(self, hidden_states, exclude_cls=True):
        B, _, dim = hidden_states.shape
        important_states = hidden_states[:, :self.merge_kept_num]
        unimportant_states = hidden_states[:, self.merge_kept_num:]

        normalized_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)
        normalized_important_states = normalized_states[:, :self.merge_kept_num]
        normalized_unimportant_states = normalized_states[:, self.merge_kept_num:]
        # [B, unimportant, dim] @ [B, dim, important] => [B, unimportant, important]
        similarity = normalized_unimportant_states @ normalized_important_states.transpose(-1, -2)
        if exclude_cls:
            similarity[..., 0] = -torch.inf

        # [B, unimportant, important] => [B, unimportant, 1]
        _, most_similar_idx = similarity.max(dim=-1, keepdim=True)

        important_states = important_states.scatter_reduce(
            dim=-2,
            index=most_similar_idx.expand(-1, -1, dim),
            src=unimportant_states,
            reduce='mean',
        )
        return important_states

# Sim model here caches result from previous frame
class ReuseModel(nn.Module):
    def __init__(
        self,
        num_frames,
        base_model_name,
        prune_kept_nums=None,
        merge_kept_nums=None,
        undo_sort=False,
        use_lora=False,
        use_sigmoid=False,
        use_softplus=False,
        lora_targets=["q_proj",'v_proj'],
        lora_rank=4,
        use_fixed_reuse=False,
        use_compressed_info=False,
        cache_dir=CACHE_DIR,
    ):
        super().__init__()

        assert num_frames >= 2, 'num_frames must be >= 2'
        self.num_frames = num_frames

        model = CLIPVisionModelWithProjection.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
        )

        if prune_kept_nums is None:
            if base_model_name == 'openai/clip-vit-large-patch14':
                prune_kept_nums = [257] * 24
            elif base_model_name == 'openai/clip-vit-base-patch16':
                prune_kept_nums = [197] * 12
            else:
                raise NotImplementedError
        if merge_kept_nums is None:
            if base_model_name == 'openai/clip-vit-large-patch14':
                merge_kept_nums = [257] * 24
            elif base_model_name == 'openai/clip-vit-base-patch16':
                merge_kept_nums = [197] * 12
            else:
                raise NotImplementedError

        kept_num = max(prune_kept_nums[0], merge_kept_nums[0])
        # Insert diffrate modules
        for layer_idx in range(len(model.vision_model.encoder.layers)):
            prune = prune_kept_nums[layer_idx]
            merge = merge_kept_nums[layer_idx]

            kept_num = min(kept_num, prune, merge)
    
            original_encoder_layer = model.vision_model.encoder.layers[layer_idx]
            model.vision_model.encoder.layers[layer_idx] = SimEncoderLayer(
                original_encoder_layer,
                prune,
                merge,
                kept_num,
                undo_sort=undo_sort,
                use_sigmoid=use_sigmoid,
                use_softplus=use_softplus,
                use_fixed_reuse=use_fixed_reuse,
                use_compressed_info=use_compressed_info
            )

        # Freeze all layers except thresholding module
        for param in model.parameters():
            param.requires_grad = False

        if use_lora:
            lora_config = LoraConfig(
                # From https://arxiv.org/pdf/2211.11733.pdf
                r=lora_rank,
                lora_alpha=4, # Set to same as r in original paper
                target_modules=lora_targets, 
                lora_dropout=0.1,
                bias="none"
            )

            model = get_peft_model(model, lora_config)

        for name, param in model.named_parameters():
            if 'threshold' in name or 'codecnet' in name:
                param.requires_grad = True

        self.model = model

        self.use_compressed_info = use_compressed_info


    def forward(
            self,
            pixel_values,
            *args,
            output_hidden_states=False,
            **kwargs,
        ):
        total_reuse_rate = 0.0

        if self.use_compressed_info:
            compressed = kwargs['compressed']
            compressed_input = compressed[:, 0]
        else:
            compressed_input = None

        output = self.model(
            pixel_values[:, 0],
            *args,
            is_first_frame=True,
            compressed_input=compressed_input,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        outputs = [output.image_embeds]
        if output_hidden_states:
            hidden_states = [output.hidden_states[-1]] # Last hidden states
        for i in range(1, self.num_frames):
            if self.use_compressed_info:
                compressed = kwargs['compressed']
                compressed_input = compressed[:, i]
            else:
                compressed_input = None
            output = self.model(
                pixel_values[:, i],
                is_first_frame=False,
                *args,
                output_hidden_states=output_hidden_states,
                compressed_input=compressed_input,
                **kwargs,
            )
            outputs.append(output.image_embeds)
            if output_hidden_states:
                hidden_states.append(output.hidden_states[-1])
            total_reuse_rate += self.get_reuse_rate()

        outputs = torch.stack(outputs, dim=1)
        if output_hidden_states:
            hidden_states = torch.stack(hidden_states, dim=1)
            outputs = (outputs, hidden_states)
        mean_reuse_rate = total_reuse_rate / (self.num_frames - 1)

        return outputs, mean_reuse_rate

    def get_reuse_rate(self):
        return self.model.get_reuse_rate()
        
        
if __name__ == '__main__':
    num_frames = 3
    reuse_model = ReuseModel(
        num_frames,
        'openai/clip-vit-base-patch16',
    )

    pixel_values = torch.randn(1, num_frames, 3, 224, 224)

    with torch.no_grad():
        _ = reuse_model(pixel_values)
