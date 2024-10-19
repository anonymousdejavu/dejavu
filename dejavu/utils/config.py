import yaml
import argparse
import warnings
from .aux import parse_lora_targets
from box import Box
import os

# Disabled options are commented out
_CONFIG_DEFAULT_TYPE = {
    'name': (None, str),
    'try_name': (None, str),
    'dataset': (None, str),
    'fps': (2, float),
    'debug': (False, bool),
    'finetune_lr': (5e-5, float),
    'restoration_lr': (5e-5, float),
    'decision_lr': (5e-6, float),
    'codecnet_lr': (5e-5, float),
    'batch_size': (4, int),
    'epochs': (1000, int),
    'base_model_name': ("openai/clip-vit-base-patch16", str),
    # Loss related
    'sloss_scaler': (1., float),
    'hloss_scaler': (0., float),
    'rloss_scaler': (1., float),
    'target_reuse_rate': (0.8, float),
    'target_similarity': (None, float),
    'rloss_pattern': ([False, False, False, True, True], list),
    'sloss_pattern': ([False, True, True, True, True], list),
    'rloss_duplicate_final_frame': (False, bool),

    'max_reuse_per_layer': (1, float),
    'seed': (42, int),
    'num_worker': (8, int),
    # 'init_threshold': -1.,
    # 'single_param': False,
    'resume_reuse_name': (None, str),
    'patience': (10, int),
    'lr_patience': (5, int),
    'lr_exponential_gamma': (0.95, float),

    # LoRA
    'use_lora': (False, bool),
    'lora_rank': (4, int),
    'lora_dropout': (0.1, float),

    # Decision Module
    'decision_type': ('threshold', str),
    'decision_mlp_inner_dim': (64, int),
    'decision_mlp_layer_pattern': (None, str),
    'decision_mlp_out_dim': (1, int),
    'decision_mlp_use_norm': (False, bool),
    'decision_mlp_dropout': (0.25, float),
    'decision_mlp_share': (False, bool),
    'decision_hyperparam': (None, float),
    'decision_reference_type': (False, bool),
    'decision_mlp_add_residual': (False, bool),
    # Gating Module
    'gating_type': ('steep_sigmoid', str),
    'gating_hyperparam': (0.25, float),
    'gating_scheduling': (False, bool),
    'gating_hard': (False, bool),
    'similarity_type': ('cosine^1', str),
    'importance_type': ('cls^1', str),
    # Restoration Module
    'restoration_type': ('passthrough', str),
    'restoration_mlp_inner_dim': (64, int),
    'restoration_mlp_disable_bias': (False, bool),
    'restoration_input_dim': (768, int),
    # 'threshold_act': '',
    # 'threshold_disable_monotonicity': False,
    # 'use_fixed_reuse': False,
    'use_hidden_states': (False, bool),
    'train_sample_rate': (None, float),
    'test_sample_rate': (None, float),
    # 'lora_targets': ['q', 'v'],
    'use_compressed_info': (False, bool),
    'use_coded_order': (False, bool),
    # 'use_onehot': False,
    # 'use_fixed_pattern': False,
    'use_cos_sim_loss': (False, bool),
    # 'ref_type': 'p4_p2n2_p1n1',
    # 'ref_space': 'global',
    'diffrate_model_name': (None, str),
    'finetune_only': (False, bool),
    'lr_scheduler': ('plateau', str),
    'num_codecnet_outputs': (1, int),
    'decision_initialize': (None, str),
    'use_gating_bn': (False, bool),

    'codecnet_channels': ([16, 32, 64] , list),
    'use_shared_codecnet': (False, bool),
    'use_codecnet_tanh': (False, bool),
    'codecnet_disable_batchnorm': (False, bool),

    # 'use_local_only': False,
    # 'use_prev_only': False,
    'use_min_cos_sim_loss': (False, bool),
    'frame_stack_pattern': ([0, 4, 8, 6, 5], list),
    'test_frame_stack_pattern': ([0, 4, 8, 6, 5], list),
    'train_dataset_step': (None, int),
    'use_start_end': (False, bool),

    'disable_final_tanh': (False, bool),
    'reference_type': ('all', str),
    'reuse_start': ('before_mlp', str),
    'augment_far_ratio': (None, float),
    'augment_same_ratio': (None, float),
    'augment_short_ratio': (None, float),
    'augment_short_far_ratio': (None, float),
    'disable_mask': (False, bool),
    'is_sweep': (False, bool),
    'skip_last_layer_reuse': (False, bool),
    'use_msrvtt_feature': (False, bool),
}

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        
    if config is None:
        config = {}

    wandb_config =  os.environ.get('WANDB_CONFIG', None)
    if wandb_config is not None:
        wandb_config = wandb_config.replace('true', 'True').replace('false', 'False')
        wandb_config = eval(wandb_config)
    else:
        wandb_config = {}

    final_config = {}
    for key, (default, type_) in _CONFIG_DEFAULT_TYPE.items():
        value = wandb_config.get(key, None)
        if value is None:
            value = config.get(key, default)
        if type_ is list and not isinstance(value, list):
            raise TypeError(f"Expected list for {key}, but got {type(value).__name__}")
        elif type_ is not type(None) and not isinstance(value, type_) and value is not None:
            try:
                final_config[key] = type_(value)
            except ValueError:
                raise TypeError(f"Type mismatch for {key}: expected {type_.__name__}, got {type(value).__name__}")
        else:
            final_config[key] = value

    assert_config(final_config)

    return Box(final_config)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # Deprecated, use load_config instead
    warnings.warn("get_args is deprecated, use load_config instead")

    parser = argparse.ArgumentParser(description='Train ReuseModel')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--try_name', type=str, help='Name of current try')
    parser.add_argument('--dataset', type=str, help='Dataset to use', choices=['how2qa', 'msrvtt'], default='how2qa')
    parser.add_argument('--fps', type=int, default=1, help='FPS for video dataset')
    parser.add_argument('--finetune_lr', type=float, default=5e-5, help='Learning rate for finetuning')
    parser.add_argument('--restoration_lr', type=float, default=5e-5, help='Learning rate for restoration module')
    parser.add_argument('--decision_lr', type=float, default=5e-6, help='Learning rate for decision module')
    parser.add_argument('--codecnet_lr', type=float, default=5e-5, help='Learning rate for codecnet')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--base_model_name', type=str, default="openai/clip-vit-base-patch16", help='Base model name')
    parser.add_argument('--sloss_scaler', type=float, default=1., help='Alpha for loss function')
    parser.add_argument('--hloss_scaler', type=float, default=0., help='Scaler for hidden states loss')
    parser.add_argument('--rloss_scaler', type=float, default=1., help='Gamma scale for sloss')
    parser.add_argument('--target_reuse_rate', type=float, default=0.8, help='Target reuse rate')
    parser.add_argument('--target_similarity', type=float, default=None, help='Target similarity')
    parser.add_argument('--num_frames', type=int, default=2, help='Number of frames to stack for training')
    # parser.add_argument('--beta', type=float, default=0.3, help='Beta for loss function')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_worker', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--init_threshold', type=float, default=-1., help='Initial threshold')
    parser.add_argument('--single_param', action='store_true', help='Use single parameter for thresholding')
    parser.add_argument('--resume_reuse_name', type=str, default=None, help='Path to checkpoint directory to resume from')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for learning rate')
    parser.add_argument('--lr_exponential_gamma', type=float, default=0.95, help='Gamma value for exponential lr scheduler')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for finetuning')
    parser.add_argument('--lora_rank', type=int, default=4, help='r in lora')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout in lora')

    parser.add_argument('--decision_type', type=str, choices=['threshold', 'mlp', 'headwise-threshold'])
    parser.add_argument('--decision_mlp_out_dim', type=int, default=1)
    parser.add_argument('--decision_mlp_use_norm', action='store_true', help='Use vector norm for mlp decision input')
    parser.add_argument('--decision_hyperparam', type=str, default=None, help='Hyperparameter for decision')

    parser.add_argument('--gating_type', type=str, choices=['gumbel', 'steep_sigmoid', 'adafuse', 'hard'], default='steep_sigmoid')
    parser.add_argument('--gating_hyperparam', type=float, default=0.25, help='Hyperparameter for gating')
    parser.add_argument('--gating_scheduling', action='store_true', help="use tau scheduling for gating")
    parser.add_argument('--gating_hard', action='store_true', help="use hard gating for training forward")
    
    parser.add_argument('--similarity_type', type=str, default='cosine^1')
    parser.add_argument('--importance_type', type=str, default='cls^1')

    parser.add_argument('--restoration_type', type=str, default='passthrough')
    parser.add_argument('--restoration_mlp_disable_bias', action='store_true', help="use bias for mlp decision unit")
    parser.add_argument('--restoration_input_dim', type=int, default=768, help='mlp restoration input dimension')


    parser.add_argument('--threshold_act', type=str, choices=['sigmoid', 'relu', 'leaky_relu', 'softplus', '', 'tanh'], default='')
    parser.add_argument('--threshold_disable_monotonicity', action='store_true', help='Disable monotonicity for thresholding')
    parser.add_argument('--use_fixed_reuse', action='store_true', help='fixed number of token reuse')
    parser.add_argument('--use_hidden_states', action='store_true', help='Use hidden states for training')
    parser.add_argument('--train_sample_rate', default=None, help='Sample rate for training dataset')
    parser.add_argument('--test_sample_rate', default=None, help='Sample rate for testing dataset')
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--lora_targets", default='q,v')
    parser.add_argument('--use_compressed_info', action='store_true', help='use compressed domain knowledge. Masked input would be given per layer')
    parser.add_argument('--use_coded_order', action='store_true', help='Make dataset to be reordered to match encoded order')
    parser.add_argument('--use_onehot', action='store_true', help='Use onehot indexing for coded orders')
    parser.add_argument('--use_fixed_pattern', action='store_true', help='Use fixed pattern for training')
    parser.add_argument('--use_cos_sim_loss', action='store_true', help='Use cosine similarity for similarity loss')
    parser.add_argument('--ref_type', type=str, choices=["p4_p2n2_p1n1", "p1_p2"], default="p4_p2n2_p1n1")
    parser.add_argument('--ref_space', type=str, choices=["1x1", "global"], default="global")
    parser.add_argument('--diffrate_model_name', type=str, default=None, help='Name of diffrate model')
    parser.add_argument('--finetune_only', action='store_true', help='Finetune only')
    parser.add_argument('--noninteractive', action='store_true', help='Noninteractive mode')
    # e_spatial_channels=[(6, 16), (16, 32), (32, 64)],
    # d_spatial_channels=[(64, 32), (64, 16), (32, 16)],
    parser.add_argument("--codecnet_channels", type=str, default="16,32,64", help="codecnet channels")
    parser.add_argument("--lr_scheduler", type=str, choices=["plateau", "exponential", "cosine"], default="plateau", help="Learning rate scheduler")
    parser.add_argument("--num_codecnet_outputs", type=int, default=1, help="Number of codecnet outputs")
    # TODO: use this flag instead of hard coded one
    parser.add_argument("--decision_initialize", type=str, default=None, choices=["adafuse"], help="decision mlp weight/bias initialization like adafuse")
    parser.add_argument("--use_gating_bn", action="store_true", help="Use batch normalization for gating")
    parser.add_argument("--use_shared_codecnet", action="store_true", help="Use shared codecnet")
    parser.add_argument("--use_codecnet_tanh", action="store_true", help="Use tanh for codecnet")
    parser.add_argument("--use_local_only", action="store_true", help="Use local only")
    parser.add_argument("--use_prev_only", action="store_true", help="Use prev only")
    parser.add_argument("--use_min_cos_sim_loss", action="store_true", help="Use min cosine similarity loss")
    parser.add_argument("--frame_stack_pattern", type=int, nargs='+', default=[0, 2, 4, 6, 5], help="Frame stack pattern")
    parser.add_argument("--train_dataset_step", type=int, default=None, help="Step for dataset")
    parser.add_argument("--rloss_pattern", type=str2bool, nargs='+', default=[False, False, False, True, True])
    parser.add_argument("--sloss_pattern", type=str2bool, nargs='+', default=[False, True, True, True, True])
    parser.add_argument("--disable_final_tanh", action="store_true", help="Disable tanh for decision")
    parser.add_argument("--reference_type", choices=['all', 'first_only'], default='all')
    parser.add_argument("--reuse_start", choices=['before_qkv', 'before_mlp'], default='before_mlp')
    parser.add_argument("--augment_far_ratio", type=float, default=None)
    parser.add_argument("--augment_same_ratio", type=float, default=None)
    parser.add_argument("--augment_short_ratio", type=float, default=None)
    parser.add_argument("--augment_short_far_ratio", type=float, default=None)
    parser.add_argument("--disable_mask", action='store_true', help="Disable ref_mask in ReuseModel")
    parser.add_argument("--skip_last_layer_reuse", action='store_true', help="Skip last layer reuse")

    args = parser.parse_args()

    assert len(args.frame_stack_pattern) == len(args.rloss_pattern), "frame_stack_pattern and rloss_pattern should have same length"
    assert len(args.frame_stack_pattern) == len(args.sloss_pattern), "frame_stack_pattern and sloss_pattern should have same length"

    if args.decision_hyperparam is not None:
        try:
            decision_hyperparam = eval(args.decision_hyperparam)
            if isinstance(decision_hyperparam, list):
                for idx, param in enumerate(decision_hyperparam):
                    decision_hyperparam[idx] = float(param)
            else:
                decision_hyperparam = float(decision_hyperparam)

            args.decision_hyperparam = decision_hyperparam
        except:
            raise ValueError(f"Invalid decision_hyperparam: {args.decision_hyperparam}")

    if args.use_compressed_info:
        channels = [int(c) for c in args.codecnet_channels.split(',')]
        e_spatial_channels = []
        prev_e = 5
        for c in channels:
            c = int(c)
            e_spatial_channels.append((prev_e, c))
            prev_e = c
        d_spatial_channels = []
        for idx, cs in enumerate(reversed(e_spatial_channels)):
            inp_c, out_c = cs
            if idx != 0:
                out_c *= 2
            if idx == len(e_spatial_channels) - 1:
                inp_c = channels[0] # 16
            d_spatial_channels.append((out_c, inp_c))
        args.e_spatial_channels = e_spatial_channels
        args.d_spatial_channels = d_spatial_channels

    if args.use_fixed_pattern:
        if args.num_frames != 5:
            print('Overriding num_frames to 5 for fixed pattern')
            args.num_frames = 5

    if args.diffrate_model_name is not None:
        print("Setting num_frames to 1 for diffrate")
        args.num_frames = 1

    if args.use_compressed_info:
        channels = [int(c) for c in args.codecnet_channels.split(',')]
        e_spatial_channels = []
        prev_e = 5
        for c in channels:
            c = int(c)
            e_spatial_channels.append((prev_e, c))
            prev_e = c
        d_spatial_channels = []
        for idx, cs in enumerate(reversed(e_spatial_channels)):
            inp_c, out_c = cs
            if idx != 0:
                out_c *= 2
            if idx == len(e_spatial_channels) - 1:
                inp_c = channels[0] # 16
            d_spatial_channels.append((out_c, inp_c))
        args.e_spatial_channels = e_spatial_channels
        args.d_spatial_channels = d_spatial_channels
    args.lora_targets = parse_lora_targets(args.lora_targets)

    assert_config(args)

    return args

def assert_config(config):
    assert len(config['frame_stack_pattern']) == len(config['rloss_pattern']), "frame_stack_pattern and rloss_pattern should have same length"
    assert len(config['frame_stack_pattern']) == len(config['sloss_pattern']), "frame_stack_pattern and sloss_pattern should have same length"

    if config['diffrate_model_name'] is not None:
        assert len(config['frame_stack_pattern']) == 1, "Diffrate model only supports num_frames=1"