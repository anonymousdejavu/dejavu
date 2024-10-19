import torch
from torch import nn
import numpy as np
from ...model.clip import CLIPVisionModelWithProjection
from ...utils.train import get_lora_state_dict
from ...model.compressed.codecnet_optim import codecnet
from ...utils import PREDEFINED_PATHS
from ...model.train import ReuseModule
from ...model.train.decision import ReuseMLP
# from peft import get_peft_model, LoraConfig

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

torch.manual_seed(42)


class ReuseEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            config,
            disable_reuse,
            original_encoder_layer,
            prune_kept_num,
            merge_kept_num,
            decision_type,
            decision_mlp_inner_dim,
            decision_mlp_add_residual,
            decision_mlp_layer_pattern,
            decision_mlp_out_dim,
            decision_mlp_use_norm,
            decision_mlp_dropout,
            decision_mlp_share,
            decision_initialize,
            decision_hyperparam,
            decision_reference_type,
            gating_type,
            gating_hyperparam,
            gating_scheduling,
            similarity_type,
            importance_type,
            restoration_type,
            restoration_mlp_inner_dim,
            restoration_mlp_disable_bias,
            restoration_input_dim,
            threshold_act='',
            threshold_disable_monotonicity=False,
            num_codecnet_outputs=1,
            use_codecnet_tanh=False,
            use_local_only=False,
            disable_final_tanh=False,
            use_compressed_info=False,
            reuse_start='before_mlp',
            disable_mask=False,
        ):
        super().__init__()

        self.config = config
        self.original_encoder_layer = original_encoder_layer

        token_num_per_side = self.config.image_size // config.patch_size
        if prune_kept_num is not None and merge_kept_num is not None:
            kept_num = min(prune_kept_num, merge_kept_num)
        else:
            kept_num = token_num_per_side**2

        if decision_mlp_share:
            assert decision_type == 'mlp', 'decision_mlp_share is only available for decision_type=mlp'
            decision_mlp = ReuseMLP(
                inner_dim=decision_mlp_inner_dim,
                add_residual=decision_mlp_add_residual,
                layer_pattern=decision_mlp_layer_pattern,
                use_compressed_info=use_compressed_info,
                decision_mlp_out_dim=decision_mlp_out_dim,
                decision_mlp_use_norm=decision_mlp_use_norm,
                decision_initialize=decision_initialize,
                decision_reference_type=decision_reference_type,
                dropout=decision_mlp_dropout,
            )
        else:
            decision_mlp = None

        self.disable_reuse = disable_reuse
        if not self.disable_reuse:
            self.reuse_module = ReuseModule(
                config.num_attention_heads,
                kept_num=kept_num,
                decision_type=decision_type,
                decision_mlp_inner_dim=decision_mlp_inner_dim,
                decision_mlp_add_residual=decision_mlp_add_residual,
                decision_mlp_layer_pattern=decision_mlp_layer_pattern,
                decision_mlp_out_dim=decision_mlp_out_dim,
                decision_mlp_use_norm=decision_mlp_use_norm,
                decision_mlp_dropout=decision_mlp_dropout,
                decision_mlp_share=decision_mlp,
                decision_initialize=decision_initialize,
                decision_hyperparam=decision_hyperparam,
                decision_reference_type=decision_reference_type,
                gating_type=gating_type,
                gating_hyperparam=gating_hyperparam,
                gating_scheduling=gating_scheduling,
                similarity_type=similarity_type,
                importance_type=importance_type,
                restoration_type=restoration_type,
                restoration_mlp_inner_dim=restoration_mlp_inner_dim,
                restoration_mlp_disable_bias=restoration_mlp_disable_bias,
                restoration_input_dim=restoration_input_dim,
                threshold_act=threshold_act,
                threshold_disable_monotonicity=threshold_disable_monotonicity,
                use_compressed_info=use_compressed_info,
                num_codecnet_outputs=num_codecnet_outputs,
                use_local_only=use_local_only,
                disable_final_tanh=disable_final_tanh,
                disable_mask=disable_mask
            )

        self.layer_norm1 = original_encoder_layer.layer_norm1
        self.self_attn = original_encoder_layer.self_attn
        self.layer_norm2 = original_encoder_layer.layer_norm2
        self.mlp = original_encoder_layer.mlp
        self.reuse_start = reuse_start

    def layer_norm1_qkv_projection(
            self,
            hidden_states,
            output_qkvs=False,
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


    def forward(
            self,
            hidden_states,
            *args,
            pre_proj=None,
            attn_weights=None,
            cached_states=None,
            output_qkvs=False,
            compressed_map=None,
            ref_mask=None,
            ref_type=None,
            **kwargs,
        ):
        bsz, N, dim = hidden_states.shape
        query_states, key_states, value_states, qkvs = self.layer_norm1_qkv_projection(hidden_states, output_qkvs=output_qkvs)

        if not self.disable_reuse and cached_states is not None:
            reuse_map, pre_proj, hidden_states, query_states, key_states, value_states = self.reuse_module(
                cached_states,
                pre_proj,
                hidden_states,
                query_states,
                key_states,
                value_states,
                attn_weights=attn_weights,
                compressed_map=compressed_map,
                ref_mask=ref_mask,
                ref_type=ref_type,
                **kwargs,
            )
        else:
            reuse_map = None

        if pre_proj is not None:
            cache_states = (pre_proj, hidden_states, query_states, key_states, value_states)
        else:
            # is first layer
            cache_states = None

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
        pre_proj = attn_output
        attn_output = self.self_attn.out_proj(attn_output)

        # Add residual
        hidden_states = hidden_states + attn_output  # [B, N, dim]

        # Before FFN: residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # After FFN: hidden_states
        hidden_states = residual + hidden_states

        if self.reuse_start == 'before_qkv':
            # In this case, reuse module will check similarity using hidden states instead of pre_proj
            return hidden_states, cache_states, hidden_states, attn_weights, qkvs, reuse_map
        elif self.reuse_start == 'before_mlp':
            return pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map
        else:
            raise NotImplementedError(f'Unknown reuse_start: {self.reuse_start}')

# Sim model here caches result from previous frame
class ReuseModel(nn.Module):
    @staticmethod
    def from_name(name, **overrides):
        config = PREDEFINED_PATHS['train'][name]
        for k, v in overrides.items():
            assert k in config, f'Unknown override key: {k}'
            config[k] = v
        return ReuseModel(**config)

    def __init__(
        self,
        base_model_name,
        decision_type='threshold',
        decision_mlp_inner_dim=64,
        decision_mlp_layer_pattern=None,
        decision_mlp_out_dim=1,
        decision_mlp_use_norm=False,
        decision_mlp_dropout=0.25,
        decision_mlp_add_residual=False,
        decision_mlp_share=False,
        decision_initialize=None,
        decision_hyperparam=None,
        gating_type='steep_sigmoid',
        gating_hyperparam=0.25,
        gating_scheduling=False,
        similarity_type='cosine^1',
        importance_type='cls^2',
        restoration_type='passthrough',
        restoration_mlp_inner_dim=64,
        restoration_mlp_disable_bias=False,
        restoration_input_dim=768,
        threshold_act='',
        disable_monotonicity=False,
        use_lora=False,
        lora_rank=4,
        lora_dropout=0.1,
        lora_targets=["q_proj",'v_proj'],
        e_spatial_channels=[(5, 16), (16, 32), (32, 64)],
        d_spatial_channels=[(64, 32), (64, 16), (32, 16)],
        dataset=None,
        resume=None,
        cache_dir=CACHE_DIR,
        num_codecnet_outputs=1,
        codecnet_disable_batchnorm=False,
        use_shared_codecnet=False,
        use_codecnet_tanh=False,
        use_local_only=False,
        use_prev_only=False,
        frame_stack_pattern=[0, 2, 2, 2, 1],
        use_coded_order=False,
        disable_final_tanh=False,
        use_compressed_info=False,
        reference_type='all',
        disable_mask=False,
        decision_reference_type=False,
        skip_last_layer_reuse=False,
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

            disable_reuse=False
            if layer_idx == 0:
                disable_reuse = True
            elif skip_last_layer_reuse and layer_idx == len(model.vision_model.encoder.layers) - 1:
                disable_reuse = True

            if isinstance(decision_hyperparam, list):
                dec_hyperparam = decision_hyperparam[layer_idx]
                assert len(decision_hyperparam) == len(model.vision_model.encoder.layers), \
                    f'Number of thresholds ({len(decision_hyperparam)}) must match the number of layers ({len(model.vision_model.encoder.layers)})'
            else:
                dec_hyperparam = decision_hyperparam

            model.vision_model.encoder.layers[layer_idx] = ReuseEncoderLayer(
                config,
                disable_reuse=disable_reuse,
                original_encoder_layer=original_encoder_layer,
                prune_kept_num=None,
                merge_kept_num=None,
                decision_type=decision_type,
                decision_mlp_inner_dim=decision_mlp_inner_dim,
                decision_mlp_add_residual=decision_mlp_add_residual,
                decision_mlp_share=decision_mlp_share,
                decision_mlp_layer_pattern=decision_mlp_layer_pattern,
                decision_mlp_out_dim=decision_mlp_out_dim,
                decision_mlp_use_norm=decision_mlp_use_norm,
                decision_mlp_dropout=decision_mlp_dropout,
                decision_initialize=decision_initialize,
                decision_hyperparam=dec_hyperparam,
                decision_reference_type=decision_reference_type,
                gating_type=gating_type,
                gating_hyperparam=gating_hyperparam,
                gating_scheduling=gating_scheduling,
                similarity_type=similarity_type,
                importance_type=importance_type,
                restoration_type=restoration_type,
                restoration_mlp_inner_dim=restoration_mlp_inner_dim,
                restoration_mlp_disable_bias=restoration_mlp_disable_bias,
                restoration_input_dim=restoration_input_dim,
                threshold_act=threshold_act,
                threshold_disable_monotonicity=disable_monotonicity,
                num_codecnet_outputs=num_codecnet_outputs,
                use_codecnet_tanh=use_codecnet_tanh,
                use_local_only=use_local_only,
                disable_final_tanh=disable_final_tanh,
                use_compressed_info=use_compressed_info,
                disable_mask=disable_mask,
            )

        # Freeze all layers except thresholding module
        for param in model.parameters():
            param.requires_grad = False

        token_num_per_side = config.image_size // config.patch_size
        # The codecnet have 6 channels
        # [mb_type, mv_x, mv_y, and three one_hot_frame_type]
        if use_shared_codecnet:
            num_hidden_layers = 1
        else:
            num_hidden_layers = config.num_hidden_layers

        self.use_coded_order = use_coded_order

        self.use_compressed_info = use_compressed_info
        if use_compressed_info:
            if decision_reference_type:
                codecnet_input_channel = 4 # [mb_type, mv_x, mv_y, qp]
            else:
                codecnet_input_channel = 4 + 3 # [mb_type, mv_x, mv_y, qp, one_hot_ref_type]

            e_spatial_channels[0] = (codecnet_input_channel, e_spatial_channels[0][1])
            self.codecnet = codecnet(
                input_shape=[codecnet_input_channel, 4, token_num_per_side, token_num_per_side],
                # Encoder
                e_spatial_channels=e_spatial_channels,
                e_spatial_kernel_size=[1, 2, 2],
                e_temporal_channels_list=[[4, 4], [4, 4], [4, 4]],
                e_activation='relu',
                e_use_bn=True,
                e_patch_per_side=token_num_per_side,
                # Decoder
                d_spatial_channels=d_spatial_channels,
                d_spatial_kernel_size=[1, 2, 2],
                num_hidden_layers=num_hidden_layers,
                num_outputs=num_codecnet_outputs,
                act='tanh' if use_codecnet_tanh else '',
                disable_batchnorm=codecnet_disable_batchnorm,
            )

        self.num_hidden_layers = num_hidden_layers
        self.num_codecnet_outputs = num_codecnet_outputs
        self.use_shared_codecnet = use_shared_codecnet

        if resume is not None:
            state_dict = get_lora_state_dict(resume)
            ret = model.load_state_dict(state_dict, strict=False)
            unexpected_unexpected_keys = list(filter(lambda k: 'lora' not in k and 'codecnet' not in k, ret.unexpected_keys))
            assert len(ret.missing_keys) == 0, f'Missing keys: {ret.missing_keys}'
            assert len(unexpected_unexpected_keys) == 0, f'Unexpected keys: {unexpected_unexpected_keys}'

            codecnet_state_dict = {}
            for k, v in state_dict.items():
                if 'codecnet' in k:
                    k = k.replace('codecnet.', '')
                    codecnet_state_dict[k] = v
            self.codecnet.load_state_dict(codecnet_state_dict)

        if use_lora:
            lora_config = LoraConfig(
                # From https://arxiv.org/pdf/2211.11733.pdf
                r=lora_rank,
                lora_alpha=4, # Set to same as r in original paper
                target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'], 
                lora_dropout=lora_dropout,
                bias="none"
            )

            model = get_peft_model(model, lora_config)

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
            *args,
            output_hidden_states=False,
            compressed=None,
            ref_mask=None,
            ref_type=None,
            **kwargs,
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
        for layer_idx, encoder_layer in enumerate(self.model.vision_model.encoder.layers):
            cached_states = None
            next_pre_proj_list = []
            next_attn_weights_list = []
            next_hidden_states_list = []
            layer_reuse_maps = []

            for frame_idx in range(self.num_frames):
                if self.use_compressed_info:
                    if self.use_shared_codecnet:
                        compressed = compressed_map[:, frame_idx, 0]
                    else:
                        compressed = compressed_map[:, frame_idx, layer_idx]
                else:
                    compressed = None

                if layer_idx == 0:
                    pre_proj = None
                    attn_weights = None
                else:
                    pre_proj = pre_proj_list[frame_idx]
                    attn_weights = attn_weights_list[frame_idx]

                if ref_mask is None:
                    r = None
                else:
                    r = ref_mask[:, frame_idx, :frame_idx]

                pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map = encoder_layer(
                    hidden_states_list[frame_idx],
                    pre_proj=pre_proj,
                    attn_weights=attn_weights,
                    cached_states=cached_states if layer_idx != 0 else None,
                    compressed_map=compressed,
                    ref_mask=r,
                    ref_type=None if ref_type is None else ref_type[:, frame_idx],
                    **kwargs,
                )

                if cache_states is not None and frame_idx < self.num_frames - 1:
                    if cached_states == None:
                        cached_states = cache_states
                    else:
                        cached_states = [torch.cat((cached, cache), dim=1) for (cached, cache) in zip(cached_states, cache_states)]

                next_pre_proj_list.append(pre_proj)
                next_attn_weights_list.append(attn_weights)
                next_hidden_states_list.append(hidden_states)

                if reuse_map is not None:
                    layer_reuse_maps.append(reuse_map)
                else:
                    layer_reuse_maps.append(torch.zeros(B, N, device=hidden_states.device))

            pre_proj_list = next_pre_proj_list
            attn_weights_list = next_attn_weights_list
            hidden_states_list = next_hidden_states_list
            if layer_idx != 0:
                layer_reuse_maps = torch.stack(layer_reuse_maps, dim=1)
                reuse_maps.append(layer_reuse_maps)

        hidden_states_list = torch.stack(hidden_states_list, dim=1)
        hidden_states_list = hidden_states_list.view(-1, N, dim)
        outputs = self.forward_post_encoder(hidden_states_list)
        outputs = outputs.view(B, F, -1)

        reuse_maps = torch.stack(reuse_maps, dim=2)

        ret = outputs, reuse_maps

        if output_hidden_states:
            ret += (hidden_states_list.reshape(B, F, N, -1),)

        return ret

    def get_reuse_rate(self):
        return self.model.get_reuse_rate()

        
        
if __name__ == '__main__':
    from ..dataset import MsrvttTrainDataset
    MODEL_NAME = 'msrvtt/try999'
    config = PREDEFINED_PATHS['train'][MODEL_NAME]
    reuse_model = ReuseModel.from_name(MODEL_NAME)

    dataset = MsrvttTrainDataset(
        pattern=config['frame_stack_pattern'],
        split='train',
        base_model_name=config['base_model_name'],
        fps=1,
        return_compressed=True,
        use_coded_order=config['use_coded_order'],
    )

    batch_size = 2
    frame_idxs = []
    pixel_values = []
    compressed = []
    ref_mask = []

    for i in range(batch_size):
        f, p, _, c, r = dataset[i]
        frame_idxs.append(f)
        pixel_values.append(p)
        compressed.append(c)
        ref_mask.append(r)

    frame_idxs = torch.stack(frame_idxs, dim=0)
    pixel_values = torch.stack(pixel_values, dim=0)
    compressed = torch.stack(compressed, dim=0)
    ref_mask = torch.stack(ref_mask, dim=0)

    for name, param in reuse_model.named_parameters():
        param.requires_grad = True
    pixel_values.requires_grad = True

    disable_mask = config.get('disable_mask', False)

    outputs, mean_reuse_rate = reuse_model(
        pixel_values,
        compressed=compressed,
        frame_idxs=frame_idxs,
        ref_mask=ref_mask if not disable_mask else None
    )

    print('outputs.shape', outputs.shape)
    print('Reuse rate:', mean_reuse_rate)

    # For debugging purpose, we will check if the gradients are properly computed
    # Only the first batch should have gradient
    loss = outputs[0].sum()
    loss.backward()

    # assert pixel_values.grad[1].sum() == 0, 'Only the first batch should have gradient'
    print('Pass')