import torch
from torch import nn
from ..train.module import ReuseModule
from ...model.clip import CLIPVisionModelWithProjection
from ...utils import PREDEFINED_PATHS

torch.manual_seed(42)

class EventfulEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            config,
            is_first_layer,
            original_encoder_layer,
            decision_type,
            decision_mlp_out_dim,
            decision_mlp_use_norm,
            decision_initialize,
            decision_hyperparam,
            gating_type,
            gating_hyperparam,
            similarity_type,
            importance_type,
            restoration_type,
            restoration_mlp_disable_bias,
            threshold_act='',
            threshold_disable_monotonicity=False,
            num_codecnet_outputs=1,
            use_local_only=False,
            disable_final_tanh=False,
            use_compressed_info=False,
        ):
        super().__init__()

        self.config = config
        self.original_encoder_layer = original_encoder_layer

        token_num_per_side = self.config.image_size // config.patch_size
        kept_num = token_num_per_side**2

        self.is_first_layer = is_first_layer
        if not self.is_first_layer:
            self.reuse_module = ReuseModule(
                config.num_attention_heads,
                kept_num=kept_num,
                decision_type=decision_type,
                decision_mlp_out_dim=decision_mlp_out_dim,
                decision_mlp_use_norm=decision_mlp_use_norm,
                decision_initialize=decision_initialize,
                decision_hyperparam=decision_hyperparam,
                gating_type=gating_type,
                gating_hyperparam=gating_hyperparam,
                similarity_type=similarity_type,
                importance_type=importance_type,
                restoration_type=restoration_type,
                restoration_mlp_disable_bias=restoration_mlp_disable_bias,
                threshold_act=threshold_act,
                threshold_disable_monotonicity=threshold_disable_monotonicity,
                use_compressed_info=use_compressed_info,
                num_codecnet_outputs=num_codecnet_outputs,
                use_local_only=use_local_only,
                disable_final_tanh=disable_final_tanh,
            )

        self.layer_norm1 = original_encoder_layer.layer_norm1
        self.self_attn = original_encoder_layer.self_attn
        self.layer_norm2 = original_encoder_layer.layer_norm2
        self.mlp = original_encoder_layer.mlp

    def qkv_projection(
            self,
            hidden_states,
            output_qkvs=False,
        ):
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


    def forward(
            self,
            hidden_states,
            attn_weights=None,
            cached_states_gate1=None,
            cached_states_gate2=None,
            cached_states_gate3=None,
            output_qkvs=False,
            compressed_map=None,
            ref_mask=None,
            **kwargs,
        ):
        bsz, N, dim = hidden_states.shape
        residual = hidden_states
        hidden_states = self.original_encoder_layer.layer_norm1(hidden_states)
        ref_states = hidden_states
        query_states, key_states, value_states, qkvs = self.qkv_projection(hidden_states, output_qkvs=output_qkvs)

        # Gate 1
        if not self.is_first_layer and cached_states_gate1 is not None:
            reuse_map, ref_states, hidden_states, query_states, key_states, value_states = self.reuse_module.forward_v2(
                cached_states_gate1,
                ref_states,
                hidden_states,
                query_states,
                key_states,
                value_states,
                attn_weights=attn_weights,
                compressed_map=compressed_map,
                ref_mask=ref_mask,
                **kwargs,
            )
        else:
            reuse_map = None

        cache_states_gate1 = (ref_states, hidden_states, query_states, key_states, value_states)

        # MHSA
        num_heads = self.self_attn.num_heads
        head_dim = self.self_attn.head_dim
        proj_shape = (bsz * num_heads, -1, head_dim)
        # [4, B, N, dim] => [4*B, N, dim]
        q = query_states * self.self_attn.scale
        q = self.self_attn._shape(q, -1, bsz)
        q = q.view(*proj_shape)

        k = self.self_attn._shape(key_states, -1, bsz)
        k = k.view(*proj_shape)

        v = self.self_attn._shape(value_states, -1, bsz)
        v = v.view(*proj_shape)

        qk = torch.bmm(q, k.transpose(1, 2))

        # attn_weights: [4*B*H, N, N]
        attn_weights = torch.nn.functional.softmax(qk, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_weights = attn_weights.view(bsz, num_heads, N, N)
        # [4*B*H, N, head_dim] => [4*B, H, N, head_dim]
        attn_output = attn_output.view(-1, self.self_attn.num_heads, N, head_dim)

        # [B, H, N, head_dim] => [B, N, H, head_dim]
        attn_output = attn_output.transpose(-3, -2)
        # [B, N, H, head_dim] => [B, N, dim(H*head_dim)]
        attn_output = attn_output.reshape(bsz, N, dim)

        ref_states = attn_output
        attn_output = self.self_attn.out_proj(attn_output)
        if not self.is_first_layer and cached_states_gate2 is not None:
            reuse_map, ref_states, attn_output = self.reuse_module.forward_v2(
                cached_states_gate2,
                ref_states,
                attn_output,
                attn_weights=attn_weights,
                compressed_map=compressed_map,
                ref_mask=ref_mask,
                **kwargs,
            )
        else:
            reuse_map = None
        cache_states_gate2 = (ref_states, attn_output)

        # Add residual
        hidden_states = residual + attn_output  # [B, N, dim]

        # Before FFN: residual
        residual = hidden_states
        ref_states = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if not self.is_first_layer and cached_states_gate3 is not None:
            reuse_map, ref_states, hidden_states = self.reuse_module.forward_v2(
                cached_states_gate3,
                ref_states,
                hidden_states,
                compressed_map=compressed_map,
                ref_mask=ref_mask,
                **kwargs,
            )
        else:
            reuse_map = None
        cache_states_gate3 = (ref_states, hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, cache_states_gate1, cache_states_gate2, cache_states_gate3, attn_weights, qkvs, reuse_map

# Sim model here caches result from previous frame
class EventfulCLIP(nn.Module):
    @staticmethod
    def from_name(name):
        config = PREDEFINED_PATHS['train'][name]
        return EventfulCLIP(**config)

    def __init__(
        self,
        base_model_name,
        decision_type='threshold',
        decision_mlp_out_dim=1,
        decision_mlp_use_norm=False,
        decision_initialize=None,
        decision_hyperparam=None,
        gating_type='steep_sigmoid',
        gating_hyperparam=0.25,
        similarity_type='cosine^1',
        importance_type='cls^2',
        restoration_type='passthrough',
        restoration_mlp_disable_bias=False,
        threshold_act='',
        disable_monotonicity=False,
        dataset=None,
        cache_dir=PREDEFINED_PATHS['root']['cache'],
        use_local_only=False,
        use_prev_only=False,
        frame_stack_pattern=[0, 2, 2, 2, 1],
        use_coded_order=False,
        disable_final_tanh=False,
        use_compressed_info=False,
        use_onehot=True,
        reference_type='all',
        **kwargs,
    ):
        '''
        This model will forward frames in the following fixed pattern
        0 (no reuse)
        2 -> 0
        4 -> 4
        6 -> 4
        5 -> 4, 6
        '''
        super().__init__()

        self.pattern = frame_stack_pattern
        self.num_frames = len(frame_stack_pattern)

        self.reference_pattern = [
            [None, None], # (0). refers to [None, None], results in [0]
            [0, None],    # (4). refers [0, None], results in [0, 4]
            [0, 1],       # (2). refers [0, 4], results in [0, 4, 2]
            [0, 2],       # (1). refers [0, 2], results in [0, 4, 2, 1]
            [1, 2],       # (3). refers [2, 4]
        ]
        self.reference_type = reference_type

        if dataset == "msrvtt":
            checkpoint_path = PREDEFINED_PATHS['msrvtt']['CLIP4Clip_checkpoint']
            model = CLIPVisionModelWithProjection.from_pretrained(
                    checkpoint_path,
                    cache_dir=cache_dir
                )
        else:
            model = CLIPVisionModelWithProjection.from_pretrained(
                base_model_name,
                cache_dir=cache_dir,
            )

        config = model.config

        # Insert diffrate modules
        for layer_idx in range(len(model.vision_model.encoder.layers)):
            if 'zerotprune' in importance_type:
                # Thus, to ensure convergence, we set the number # of iterations to 30-50, 5-10, and 1
                #  in the first three layers, medium layers, and last three layers, respectively.
                if layer_idx < 3:
                    importance_type = 'zerotprune@30'
                elif layer_idx > len(model.vision_model.encoder.layers) - 3:
                    importance_type = 'zerotprune@1'
                else:
                    importance_type = 'zerotprune@5'

            original_encoder_layer = model.vision_model.encoder.layers[layer_idx]
            model.vision_model.encoder.layers[layer_idx] = EventfulEncoderLayer(
                config,
                is_first_layer=layer_idx == 0,
                original_encoder_layer=original_encoder_layer,
                decision_type=decision_type,
                decision_mlp_out_dim=decision_mlp_out_dim,
                decision_mlp_use_norm=decision_mlp_use_norm,
                decision_initialize=decision_initialize,
                decision_hyperparam=decision_hyperparam,
                gating_type=gating_type,
                gating_hyperparam=gating_hyperparam,
                similarity_type=similarity_type,
                importance_type=importance_type,
                restoration_type=restoration_type,
                restoration_mlp_disable_bias=restoration_mlp_disable_bias,
                threshold_act=threshold_act,
                threshold_disable_monotonicity=disable_monotonicity,
                use_local_only=use_local_only,
                disable_final_tanh=disable_final_tanh,
            )

        # Freeze all layers except thresholding module
        for param in model.parameters():
            param.requires_grad = False

        self.use_coded_order = use_coded_order
        self.use_onehot = use_onehot

        self.use_compressed_info = use_compressed_info

        for name, param in model.named_parameters():
            if 'reuse_module' in name or 'codecnet' in name:
                param.requires_grad = True

        self.model = model

        self.use_prev_only = use_prev_only


    def forward_pre_encoder(
            self,
            pixel_values
        ):
        hidden_states = self.model.vision_model.embeddings(pixel_values)
        hidden_states = self.model.vision_model.pre_layrnorm(hidden_states)

        return hidden_states

    def forward_post_encoder(
            self,
            hidden_states,
        ):
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.model.vision_model.post_layernorm(pooled_output)
        image_embeds = self.model.visual_projection(pooled_output)

        return image_embeds

    def forward(
            self,
            pixel_values,
            compressed=None,
            ref_mask=None,
            cached_states_from_prev_batch_gate1=None,
            cached_states_from_prev_batch_gate2=None,
            cached_states_from_prev_batch_gate3=None,
        ):
        assert pixel_values.shape[1] == self.num_frames, f'{self.num_frames} frames must be given'

        B, F, *_ = pixel_values.shape
        pixel_values = pixel_values.view(-1, 3, 224, 224)
        hidden_states_list = self.forward_pre_encoder(pixel_values)
        _, N, dim = hidden_states_list.shape
        hidden_states_list = hidden_states_list.view(B, -1, N, dim)
        # [B, F, N, dim] => [F, B, N, dim]
        hidden_states_list = hidden_states_list.transpose(0, 1)
        
        if self.use_compressed_info:
            B, F, C, T, H, W = compressed.shape
            compressed_input = compressed.view(B*F, -1, T, H, W)
            compressed_map = self.codecnet(compressed_input)
            compressed_map = compressed_map.view(B, F, self.num_hidden_layers, self.num_codecnet_outputs, -1)

        # (phqkv, B, frame_idx - 1, N, dim)

        reuse_maps = []
        cached_states_for_next_batch_gate1 = []
        cached_states_for_next_batch_gate2 = []
        cached_states_for_next_batch_gate3 = []
        for layer_idx, encoder_layer in enumerate(self.model.vision_model.encoder.layers):
            next_attn_weights_list = []
            next_hidden_states_list = []
            layer_reuse_maps = []

            if cached_states_from_prev_batch_gate1 is not None:
                cached_states_gate1 = cached_states_from_prev_batch_gate1[layer_idx]
                cached_states_gate2 = cached_states_from_prev_batch_gate2[layer_idx]
                cached_states_gate3 = cached_states_from_prev_batch_gate3[layer_idx]
            else:
                cached_states_gate1 = None
                cached_states_gate2 = None
                cached_states_gate3 = None

            for frame_idx in range(self.num_frames):
                if layer_idx == 0:
                    attn_weights = None
                else:
                    attn_weights = attn_weights_list[frame_idx]

                if ref_mask is None:
                    r = None
                else:
                    r = ref_mask[:, frame_idx, :frame_idx]

                (
                    hidden_states,
                    cache_states_gate1,
                    cache_states_gate2,
                    cache_states_gate3,
                    attn_weights,
                    qkvs,
                    reuse_map
                ) = encoder_layer(
                    hidden_states_list[frame_idx],
                    attn_weights=attn_weights,
                    cached_states_gate1=cached_states_gate1 if layer_idx != 0 else None,
                    cached_states_gate2=cached_states_gate2 if layer_idx != 0 else None,
                    cached_states_gate3=cached_states_gate3 if layer_idx != 0 else None,
                    ref_mask=r,
                )

                if cache_states_gate1 is not None and frame_idx < self.num_frames:
                    cached_states_gate1 = cache_states_gate1
                    cached_states_gate2 = cache_states_gate2
                    cached_states_gate3 = cache_states_gate3

                next_attn_weights_list.append(attn_weights)
                next_hidden_states_list.append(hidden_states)

                if reuse_map is not None:
                    layer_reuse_maps.append(reuse_map)
                else:
                    layer_reuse_maps.append(torch.zeros(B, N, device=hidden_states.device))

            attn_weights_list = next_attn_weights_list
            hidden_states_list = next_hidden_states_list
            if layer_idx != 0:
                layer_reuse_maps = torch.stack(layer_reuse_maps, dim=1)
                reuse_maps.append(layer_reuse_maps)

            cached_states_for_next_batch_gate1.append(cached_states_gate1)
            cached_states_for_next_batch_gate2.append(cached_states_gate2)
            cached_states_for_next_batch_gate3.append(cached_states_gate3)


        hidden_states_list = torch.stack(hidden_states_list, dim=1)
        hidden_states_list = hidden_states_list.view(-1, N, dim)
        outputs = self.forward_post_encoder(hidden_states_list)
        outputs = outputs.view(B, F, -1)

        reuse_maps = torch.stack(reuse_maps, dim=2)

        return outputs, reuse_maps, cached_states_for_next_batch_gate1, cached_states_for_next_batch_gate2, cached_states_for_next_batch_gate3

