PREDEFINED_PATHS = {
    'root': {
        'data': '/mnt',
        'config': '/workspace/config',
        'cache': '/mnt/ssd1/cache',
    },
    'how2qa': {
        'video': '/mnt/hdd2/how2qa/video',
        'video_transcoded': '/mnt/ssd3/dataset/how2qa/video_transcoded',
        'compressed': '/mnt/ssd3/dataset/how2qa/compressed_domain',
        'diffrate_dir': '/mnt/ssd4/dataset/how2qa/diffrate',
        'is_clipped_video': False,
        'feature_dir': '/mnt/ssd4/dataset/how2qa/feature',
        'reuse_dir': '/mnt/ssd4/dataset/how2qa/reuse',
        'reuse_exp_dir': '/mnt/ssd3/dataset/how2qa/reuse_exp',
        'reuse_trace_dir': '/mnt/ssd2/dataset/how2qa/reuse_trace',
        'split': {
            'train': '/workspace/dataset/how2QA_train_release.csv',
            'test': '/workspace/dataset/how2QA_test_public_release.csv',
            'frozenbilm': '/workspace/dataset/public_val.csv',
        }
    },
    'msrvtt': {
        'video': '/mnt/hdd2/MSRVTT/msrvtt_data/MSRVTT_Videos',
        'video_transcoded': '/mnt/ssd2/dataset/msrvtt/video_transcoded',
        'compressed': '/mnt/ssd3/dataset/msrvtt/compressed_domain_1fps',
        'diffrate_dir': '/mnt/ssd4/dataset/msrvtt/diffrate',
        'is_clipped_video': False,
        'feature_dir': '/mnt/ssd2/dataset/msrvtt/feature',
        'reuse_dir': '/mnt/ssd2/dataset/msrvtt/reuse',
        'reuse_trace_dir': '/mnt/ssd2/dataset/msrvtt/reuse_trace',
        'CLIP4Clip_checkpoint': '/mnt/ssd3/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/best_hf_model',
        'split': {
            'train': '/workspace/dataset/MSRVTT_train.9k.csv',
            'test': '/workspace/dataset/MSRVTT_JSFUSION_test.csv',
        }
    },
    'train': {
        'checkpoint_dir': '/mnt/nfs/',
        'diffrate_dir': '/mnt/ssd3/diffrate',
    },
}

from pathlib import Path
from yaml import safe_load
import warnings
# recursively iterate dict and upon finding '/mnt/' in any value, replace it with '/mnt/nfs'
def replace_recursively(d, src, tgt):
    for k, v in d.items():
        if isinstance(v, dict):
            replace_recursively(v, src, tgt)
        elif isinstance(v, str):
            d[k] = v.replace(src, tgt)

config_path = Path(PREDEFINED_PATHS['root']['config'])

VERBOSE=False
import warnings
for file_path in config_path.rglob('*'):
    if file_path.is_file() and file_path.suffix == '.yaml':
        with open(file_path, 'r') as f:
            try:
                config = safe_load(f)
                name = f"{config['name']}/{config['try_name']}"
                PREDEFINED_PATHS['train'][name] = config
            except KeyError:
                if VERBOSE:
                    warnings.warn(f"Invalid config file: {file_path}")
