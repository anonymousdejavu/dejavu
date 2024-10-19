import torch
from pathlib import Path

from . import PREDEFINED_PATHS
from .aux import rename_base_model
from ..utils.aux import parse_lora_targets
from torch.nn import functional as F
from torch import nn
import argparse

import warnings
warnings.simplefilter('always', DeprecationWarning)

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

def get_checkpoint_path(model_name, epoch=None):
    checkpoint_dir = Path(PREDEFINED_PATHS['train']['checkpoint_dir'])
    base_model_renamed = rename_base_model(PREDEFINED_PATHS['train'][model_name]['base_model_name'])
    if epoch is None:
        epoch = 'best'
    else:
        epoch = f'epoch_{epoch}'

    # Check safetensor first
    checkpoint_path = checkpoint_dir / base_model_renamed / model_name / 'checkpoints' / epoch / 'model.safetensors'
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path = checkpoint_dir / base_model_renamed / model_name / 'checkpoints' / epoch / 'pytorch_model.bin'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path

def get_monotonic_threshold(
        non_monotonic_threshold,
        act='',
        disable_monotonicity=False
    ):
    num_activation = 0

    if act == 'sigmoid':
        sim_threshold = torch.sigmoid(non_monotonic_threshold)
        num_activation += 1
    elif act == 'softplus':
        sim_threshold = F.softplus(non_monotonic_threshold)
        num_activation += 1
    elif act == 'relu':
        sim_threshold = F.relu(non_monotonic_threshold)
        num_activation += 1
    elif act == 'leaky_relu':
        sim_threshold = F.leaky_relu(non_monotonic_threshold)
        num_activation += 1
    elif act == 'tanh':
        sim_threshold = F.tanh(non_monotonic_threshold)
        num_activation += 1
    elif act == '':
        sim_threshold = non_monotonic_threshold
    else:
        raise NotImplementedError
    
    assert num_activation <= 1, "Only one activation function can be used"

    if disable_monotonicity:
        return sim_threshold

    # In order to force threshold to be monotonically decreasing
    cumsum = torch.cumsum(sim_threshold, dim=-1)
    monotonic = 1. - cumsum 

    return monotonic
    

def get_thresholds(
        model_name,
        checkpoint_path=None,
        epoch=None,
    ):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name, epoch=epoch)
    state_dict = load_state_dict(checkpoint_path)

    key = 'threshold'
    thresholds_per_layer = []

    config = PREDEFINED_PATHS['train'][model_name]
    act = config.get('act', '')
    disable_monotonicity = config.get('disable_monotonicity', False)
    for k, v in state_dict.items():
        if key in k:
            monotonic = get_monotonic_threshold(v, act, disable_monotonicity=disable_monotonicity)
            thresholds_per_layer.append(monotonic)

    return thresholds_per_layer

def get_reuse_map(model_name, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    reuse_map_per_layer = []
    for k, v in state_dict.items():
        if 'reuse_map' in k:
            reuse_map = torch.empty_like(v)
            reuse_map[torch.sigmoid(40*v)>0.5] = True
            reuse_map[torch.sigmoid(40*v)<=0.5] = False
            reuse_map_per_layer.append(reuse_map.bool())
    return reuse_map_per_layer

def get_lora_path(model_name):
    lora_dir = Path(PREDEFINED_PATHS['train']['lora_dir'])
    base_model_renamed = rename_base_model(PREDEFINED_PATHS['train'][model_name]['base_model_name'])
    return lora_dir / base_model_renamed / model_name / 'checkpoints/best/pytorch_model.bin'


def binary_reuse_map(reuse_map):
    result = torch.empty_like(reuse_map)
    result[torch.sigmoid(40*reuse_map)>0.5] = True
    result[torch.sigmoid(40*reuse_map)<=0.5] = False
    return result.bool()

def load_state_dict(checkpoint_path):
    from safetensors import safe_open 
    checkpoint_name = str(checkpoint_path)
    if 'safetensor' in checkpoint_name:
        state_dict = {}
        with safe_open(checkpoint_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    return state_dict


def get_lora_state_dict(
        model_name,
        epoch=None,
    ):
    assert PREDEFINED_PATHS['train'][model_name]['use_lora'], 'This model is not trained with LoRA'
    checkpoint_path = get_checkpoint_path(model_name, epoch=epoch)
    state_dict = load_state_dict(checkpoint_path)

    skip_strs = ['lora', 'reuse', 'codecnet']

    lora_state_dict = {}
    for k, v in state_dict.items():
        if any(skip_str in k for skip_str in skip_strs):
            continue

        k = k.replace('model.base_model.model.', '')
        k = k.replace('base_layer.', '')
        k = k.replace('original_encoder_layer.', 'original_layer.')
        lora_state_dict[k] = v

    return lora_state_dict

   
def get_diffrate_lora_merged_state_dict(
        model,
        model_name,
        checkpoint_path=None,
        reuse_converted=True,
        cache_dir=CACHE_DIR
    ):
    from peft import LoraConfig, get_peft_model
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)
        lora_path = get_lora_path(model_name)
    try:
        # reuse lora state_dict if checkpoint_path is older
        if reuse_converted and checkpoint_path.stat().st_mtime < lora_path.stat().st_mtime:
            state_dict = torch.load(lora_path, map_location='cpu')
            return state_dict
    except:
        pass

    print('Converting DIFFRATE LoRA merged state_dict')
    # state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = load_state_dict(checkpoint_path)
    
    lora_rank = PREDEFINED_PATHS['train'][model_name].get('lora_rank', 4)
    lora_targets = PREDEFINED_PATHS['train'][model_name].get('lora_targets', 'q,v')

    lora_targets = parse_lora_targets(lora_targets)
    
    lora_config = LoraConfig(
        # From https://arxiv.org/pdf/2211.11733.pdf
        r=lora_rank,
        lora_alpha=4, # Set to same as r in original paper
        target_modules=lora_targets, 
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    
    renamed_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        renamed_state_dict[k] = v

    ret = model.load_state_dict(renamed_state_dict, strict=False)
    assert len(ret.unexpected_keys) == 0

    model = model.merge_and_unload()
    merged_state_dict = model.state_dict()

    renamed_state_dict = {}
    for k, v in merged_state_dict.items():
        k = k.replace('base_model.model.', '') 
        renamed_state_dict[k] = v

    lora_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(renamed_state_dict, lora_path)

    return renamed_state_dict

def get_codecnet_state_dict(reuse_model_name, epoch=None, checkpoint_path=None):
    # prev_keys and new_keys is for ported codecnet.contracting
    prev_keys = ["0.0.weight", "0.0.bias", "1.weight", "1.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked", "3.inner.0.weight", "3.inner.3.weight"]
    new_keys = ["0.weight", "0.bias", "3.weight", "3.bias", "3.running_mean", "3.running_var", "3.num_batches_tracked", "5.inner.0.weight", "5.inner.3.weight"]

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(reuse_model_name, epoch=epoch)
    state_dict = load_state_dict(checkpoint_path)

    codecnet_dict = {}
    for k, v in state_dict.items():
        if 'codecnet' in k:
            k = k.replace('_orig_mod.model.base_model.model.', '')
            k = k.replace('model.base_model.model.', '') # this line is to port optimized-sim
            k = k.replace('codecnet.', '')
            codecnet_dict[k] = v

    return codecnet_dict

def load_dataset(args):
    from ..train.dataset import How2qaTrainDataset, MsrvttTrainDataset
    # Load Dataset
    if args.dataset == 'how2qa':
        train_dataset = How2qaTrainDataset(
            num_frames=args.num_frames,
            split='test',
            base_model_name=args.base_model_name,
            fps=args.fps,
            return_compressed=args.use_compressed_info,
        )
        test_dataset = How2qaTrainDataset(
            num_frames=args.num_frames,
            split='frozenbilm',
            base_model_name=args.base_model_name,
            fps=args.fps,
            return_compressed=args.use_compressed_info,
        )
    elif args.dataset == 'msrvtt':
        train_dataset = MsrvttTrainDataset(
            num_frames=args.num_frames,
            split='train',
            base_model_name=args.base_model_name,
            fps=args.fps,
            return_compressed=args.use_compressed_info,
        )
        test_dataset = MsrvttTrainDataset(
            num_frames=args.num_frames,
            split='test',
            base_model_name=args.base_model_name,
            fps=args.fps,
            return_compressed=args.use_compressed_info,
        )
        exit(0)
    else:
        raise NotImplementedError
    
    return train_dataset, test_dataset


def init_threshold(threshold, act, disable_monotonicity=False):
    if disable_monotonicity:
        if act in ['sigmoid', '', 'tanh']:
            nn.init.constant_(threshold, 0)
        else:
            raise NotImplementedError(f"act={act} is not supported with disable_monotonicity=True")
    else:
        if act in ['', 'relu', 'leaky_relu']:
            nn.init.constant_(threshold, 0.01)
        elif act in ['sigmoid', 'softplus']:
            # roughly 0.01
            nn.init.constant_(threshold, -4.5)
        else:
            raise NotImplementedError(f"act={act} is not supported with disable_monotonicity=False")

def normalize_vector(v, return_norm=False):
    norm = v.norm(dim=-1, keepdim=True)
    ret = v / (norm + 1e-7)
    if return_norm:
        return ret, norm
    return ret


def reuse_loss(
        output,
        original_hidden_states,
        original_output,
        reuse_maps,
        target_reuse_rate,
        target_similarity,
        sloss_scaler=0.8,
        hloss_scaler=0.8,
        rloss_scaler=1.0,
        use_cos_sim_loss=False,
        use_min_cos_sim_loss=False,
        rloss_pattern=None,
        sloss_pattern=None,
    ):
    if original_hidden_states is not None:
        hidden_states = output[1]
        output = output[0]
        cos_sim = F.cosine_similarity(hidden_states, original_hidden_states, dim=-1)
        if sloss_pattern is not None:
            hloss = hloss[:, sloss_pattern]
    else:
        hloss = 0

    cos_sim = F.cosine_similarity(output, original_output, dim=-1)
    if sloss_pattern is not None:
        cos_sim = cos_sim[:, sloss_pattern]
    if rloss_pattern is not None:
        reuse_maps = reuse_maps[:, rloss_pattern]

    cos_error = 1 - cos_sim.mean()

    mse = torch.nn.MSELoss(reduction='none')(output, original_output).mean(dim=-1) 
    mse_sloss = mse.max(dim=-1)[0].mean()

    reuse_maps = reuse_maps.mean()

    if use_min_cos_sim_loss:
        cos_sloss = 1 - cos_sim.min(dim=-1)[0].mean()
    else:
        cos_sloss = cos_error

    if target_similarity is not None:
        assert use_cos_sim_loss, "target_similarity is only valid when use_cos_sim_loss is True"
        target_sloss = 1. - target_similarity
        cos_sloss = torch.relu(cos_sloss - target_sloss)
        rloss = 1 - reuse_maps
    else:
        rloss = torch.relu(target_reuse_rate - reuse_maps)

    if use_cos_sim_loss:
        sloss = cos_sloss
    else:
        sloss = mse_sloss

    loss = sloss_scaler*sloss + hloss_scaler*hloss + rloss_scaler*rloss

    return loss, cos_error, mse_sloss, reuse_maps

def reuse_loss_v2(
    hidden_states,
    output,
    original_hidden_states,
    original_output,
    reuse_maps,
    target_reuse_rate,
    sloss_pattern=None,
    rloss_pattern=None,
    use_min_hloss=False,
    use_min_sloss=True,
    hloss_scaler=0.8,
    sloss_scaler=0.8,
    rloss_scaler=1.0,
    max_reuse_per_layer=1,
    rloss_duplicate_final_frame=False,
):
    '''Drop support for mse loss and target similarity'''
    if sloss_pattern is not None:
        hidden_states = hidden_states[:, sloss_pattern]
        output = output[:, sloss_pattern]
        original_hidden_states = original_hidden_states[:, sloss_pattern]
        original_output = original_output[:, sloss_pattern]
    if rloss_pattern is not None:
        reuse_maps = reuse_maps[:, rloss_pattern]

    # Calculate mean cosine similarities of hidden states
    hidden_states_sim = F.cosine_similarity(hidden_states, original_hidden_states, dim=-1)

    # Calculate mean cosine similarities of output
    output_sim = F.cosine_similarity(output, original_output, dim=-1)

    # [bsz, num_stack, num_layers, num_tokens]
    reuse_rate_per_frame = reuse_maps.mean(dim=(-1, -2))

    reuse_rate_per_layer = reuse_maps.mean(dim=-1)
    reuse_rate_per_layer = torch.clamp(reuse_rate_per_layer, max=max_reuse_per_layer)
    reuse_rate_sum = reuse_rate_per_layer.sum()
    reuse_rate_numel = reuse_rate_per_layer.numel()

    if rloss_duplicate_final_frame:
        reuse_rate_sum += reuse_rate_per_frame[..., -1].sum()
        reuse_rate_numel += reuse_rate_per_frame[..., -1].numel()

    reuse_rate = reuse_rate_sum / reuse_rate_numel

    if use_min_hloss:
        hloss = 1 - hidden_states_sim.min(dim=1)[0].mean()
    else:
        hloss = 1 - hidden_states_sim.mean()

    if use_min_sloss:
        sloss = 1 - output_sim.min(dim=-1)[0].mean()
    else:
        sloss = 1 - output_sim.mean()

    rloss = torch.relu(target_reuse_rate - reuse_rate)

    hidden_error = 1 - hidden_states_sim.mean()
    cls_error = 1 - output_sim.mean()

    loss = sloss_scaler*sloss + hloss_scaler*hloss + rloss_scaler*rloss

    return loss, hidden_error, cls_error, reuse_rate, reuse_rate_per_frame

def get_state_dict(model_name, epoch=None, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name, epoch=epoch)
    state_dict = load_state_dict(checkpoint_path)
    return state_dict


def get_reuse_module_state_dict(reuse_model_name, epoch=None, checkpoint_path=None):
    state_dict = get_state_dict(reuse_model_name, epoch=epoch, checkpoint_path=checkpoint_path)
    d = {}
    for k, v in state_dict.items():
        if not 'reuse_module' in k:
            continue

        k = k.replace('model.base_model.model.', '')
        k = k.replace('model.vision_model.', 'vision_model.')
        k = k.replace('reuse_module.', '')

        if 'decision_module' in k:
            # We should lower the layer number by 0
            # vision_model.encoder.layers.0.decision_module.bn.weight
            splitted = k.split('.')
            prefix = splitted[:3]
            layer_num = int(splitted[3])
            suffix = splitted[4:]
            new_layer_num = layer_num - 1

            k = '.'.join(prefix + [str(new_layer_num)] + suffix)

        d[k] = v
    return d
