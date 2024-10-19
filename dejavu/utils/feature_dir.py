from .dataset import get_feature_dir
from .preprocess import is_integer

def get_feature_dir_reuse(dataset_name, base_model_name, fps, split, reuse_model_name, refresh_interval=0):
    diffrate_root = get_feature_dir(dataset_name, base_model_name, fps, split, 'reuse_dir')

    reuse_dir = diffrate_root / reuse_model_name
    if refresh_interval != 0:
        reuse_dir /= f'interval_{refresh_interval}'

    return reuse_dir

def get_feature_dir_cmc(dataset_name, base_model_name, fps, split, threshold, reuse_start_before_mlp=False):
    feature_dir_root = get_feature_dir(dataset_name, base_model_name, fps, split, 'reuse_dir')
    if isinstance(threshold, list):
        threshold_int = []
        for t in threshold:
            if is_integer(t):
                threshold_int.append(str(int(t)))
            else:
                threshold_int.append(str(t))
        threshold_str = '_'.join(threshold_int)
    elif is_integer(threshold):
        threshold_str = str(int(threshold))
    else:
        threshold_str = str(threshold)

    if reuse_start_before_mlp:
        return feature_dir_root / f'cmc_{threshold_str}_before_mlp'
    else:
        return feature_dir_root / f'cmc_{threshold_str}'

def get_feature_dir_eventful(dataset_name, base_model_name, fps, split, top_r):
    feature_dir_root = get_feature_dir(dataset_name, base_model_name, fps, split, 'reuse_dir')

    return feature_dir_root / f'eventful_{top_r}'