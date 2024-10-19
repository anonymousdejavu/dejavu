from pathlib import Path
from . import PREDEFINED_PATHS, get_feature_dir
import re

def get_log_path(dataset, name):
    diffrate_dir = Path(PREDEFINED_PATHS['train']['diffrate_dir'])
    if 'original' in name: # In the form of original-8.7
        flops = name.split('-')[-1]
    elif 'diffrate' in name:
        flops = name.split('/')[-1]
    else:
        raise NotImplementedError

    log_path = diffrate_dir / dataset / flops / 'log_rank0.txt'
    return log_path

def get_diffrate_prune_merge(dataset, name, epoch=None):
    log_path = get_log_path(dataset, name)

    with open(log_path, 'r') as f:
        for line in f:
            if 'INFO prune kept number:' in line:
                prune_txt = line.strip()
            elif 'INFO merge kept number:' in line:
                merge_txt = line.strip()
            if epoch is not None and f'Epoch: [{epoch}]' in line:
                break

    # Parse the list from the following pattern
    # [2024-02-06 23:47:12 root] (engine.py 118): INFO prune kept number:[197, 193, 169, 155, 126, 106, 103, 98, 87, 73, 60, 5]
    # [2024-02-06 23:47:12 root] (engine.py 119): INFO merge kept number:[197, 175, 160, 148, 107, 103, 101, 91, 79, 65, 57, 5]
    # [2024-02-06 23:47:11 root] (utils.py 286): INFO Epoch: [299]  [259/260]
    pat = re.compile(r'.*\[(.*)\]')

    prune = pat.match(prune_txt).group(1)
    merge = pat.match(merge_txt).group(1)

    prune = list(map(int, prune.split(', ')))
    merge = list(map(int, merge.split(', ')))

    return prune, merge

def get_feature_dir_diffrate(dataset_name, base_model_name, fps, split, diffrate_model_name):
    diffrate_root = get_feature_dir(dataset_name, base_model_name, fps, split, dir_key='diffrate_dir')
    feature_dir = Path(diffrate_root) / diffrate_model_name
    return feature_dir