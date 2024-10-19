import torch
from dejavu.train.reuse.model import ReuseModel as TrainModel
from dejavu.utils.train import load_state_dict, get_checkpoint_path
from . import PREDEFINED_PATHS


def initialize_train_model(reuse_model_name, epoch, use_gumbel=False):
    config = PREDEFINED_PATHS['train'][reuse_model_name]
    if not use_gumbel:
        config['gating_type'] = 'hard'
    train_model = TrainModel(**config)
    checkpoint_path = get_checkpoint_path(reuse_model_name, epoch=epoch)
    state_dict = load_state_dict(checkpoint_path)
    ret = train_model.load_state_dict(state_dict, strict=False)
    train_model.num_frames = train_model.num_frames - 1
    assert len(ret.unexpected_keys) == 0

    return train_model

def run_batch_for_train_model(
        model,
        pixel_values,
        compressed=None,
        cached_states_from_prev_batch=None,
        ref_mask=None,
        tau=None,
    ):
    num_frames = model.num_frames

    assert pixel_values.shape[1] == num_frames, f'Expected {num_frames} frames, got {pixel_values.shape[1]}'

    '''Adapted from ReuseModel.forward'''
    B, F, *_ = pixel_values.shape
    pixel_values = pixel_values.view(-1, 3, 224, 224)
    hidden_states_list = model.forward_pre_encoder(pixel_values)
    _, N, dim = hidden_states_list.shape
    hidden_states_list = hidden_states_list.view(B, -1, N, dim)
    # [B, F, N, dim] => [F, B, N, dim]
    hidden_states_list = hidden_states_list.transpose(0, 1)
    
    if model.use_compressed_info:
        B, F, C, T, H, W = compressed.shape
        compressed_input = compressed.view(B*F, -1, T, H, W)
        compressed_map = model.codecnet(compressed_input)
        compressed_map = compressed_map.view(B, F, model.num_hidden_layers, model.num_codecnet_outputs, -1)

    # (phqkv, B, frame_idx - 1, N, dim)
    reuse_maps = []
    cached_states_for_next_batch = []
    for layer_idx, encoder_layer in enumerate(model.model.vision_model.encoder.layers):
        next_pre_proj_list = []
        next_attn_weights_list = []
        next_hidden_states_list = []
        layer_reuse_maps = []
        if cached_states_from_prev_batch is not None:
            cached_states = cached_states_from_prev_batch[layer_idx]
        else:
            cached_states = None

        for frame_idx in range(num_frames):
            if model.use_compressed_info:
                if model.use_shared_codecnet:
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
                r = ref_mask[:, frame_idx, :frame_idx+1]

            pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map = encoder_layer(
                hidden_states_list[frame_idx],
                pre_proj=pre_proj,
                attn_weights=attn_weights,
                cached_states=cached_states if layer_idx != 0 else None,
                compressed_map=compressed,
                ref_mask=r,
                hard=True,
                tau=tau,
            )

            if cache_states is not None and frame_idx < num_frames:
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

        cached_states_for_next_batch.append(cached_states)
    

    hidden_states_list = torch.stack(hidden_states_list, dim=1)
    hidden_states_list = hidden_states_list.view(-1, N, dim)
    outputs = model.forward_post_encoder(hidden_states_list)
    outputs = outputs.view(B, F, -1)

    reuse_maps = torch.stack(reuse_maps, dim=2)

    return outputs, reuse_maps, cached_states_for_next_batch
