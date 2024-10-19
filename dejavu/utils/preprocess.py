from .predefined_paths import PREDEFINED_PATHS
from pathlib import Path
import numpy as np
from .aux import rename_base_model

def is_integer(value):
    try:
        int_value = int(value)
        float_value = float(value)
        if float_value == int_value:
            return True
        else:
            return False
    except ValueError:
        return False

def get_available_datasets():
    filter_names = ['train', 'throughput', 'root']
    return list(filter(lambda k: k not in filter_names, PREDEFINED_PATHS.keys()))

def get_sanitized_path(dataset, split):
    original = Path(PREDEFINED_PATHS[dataset]['split'][split])
    sanitized_name = f'{original.stem}_sanitized{original.suffix}'
    return original.parent / sanitized_name

def get_available_splits(dataset='all', exclude_sanitized=False):
    if dataset != 'all':
        splits = PREDEFINED_PATHS[dataset]['split'].keys()
    else:
        splits = []
        datasets = get_available_datasets()
        for d in datasets:
            splits.extend(PREDEFINED_PATHS[d]['split'].keys())
    if exclude_sanitized:
        splits = list(filter(lambda k: 'sanitized' not in k, splits))

    splits = list(np.unique(splits))
    return splits

def filter_existing_files(output_paths, *others):
    output_paths_len = len(output_paths)
    assert all(len(o) == output_paths_len for o in others), "All lists must have the same length"

    output_paths_filtered = []
    others_filtered = [[] for _ in range(len(others))]

    num_skipped = 0
    for i in range(output_paths_len):
        output_path = Path(output_paths[i])
        if not output_path.exists():
            output_paths_filtered.append(output_path)
            for j, o in enumerate(others):
                others_filtered[j].append(o[i])
        else:
            num_skipped += 1

    print(f"Skipping {num_skipped} existing files out of {output_paths_len}")

    ret = (output_paths_filtered, *others_filtered)

    return ret

def get_feature_path(feature_dir, video_id, feature_type, frame_num, use_v2=False):
    assert feature_type in ['i', 'p', 'h', 'hh', 'o', 'c'], f"Invalid feature type {feature_type} (must be one of 'i', 'p', 'h', 'hh', 'o', 'c')"

    if use_v2:
        return feature_dir / f'{video_id}' / f'{feature_type}_{frame_num}.npz'
    else:
        return feature_dir / f'{video_id}_{feature_type}_{frame_num}.npz'

def get_coded_order_path(feature_dir, video_id):
    return feature_dir / f'{video_id}_coded_order.npz'

def get_transcode_dir(dataset, fps, base_model_name=None):
    ret = Path(PREDEFINED_PATHS[dataset]['video_transcoded'])

    if base_model_name is not None:
        base_model_name = rename_base_model(base_model_name)
        ret /= base_model_name
    # To refrain from naming file name 1.0
    if is_integer(fps):
        fps = int(fps)
    ret /= f"{fps}fps"
    return ret

def split_per_rank(rank, world_size, *items):
    assert world_size > 0, "World size must be positive"
    assert rank >= 0 and rank < world_size, f"Rank must be in [0, {world_size})"
    assert len(items) > 0, "At least one item must be provided"
    assert all(len(items[0]) == len(i) for i in items), "All items must have the same length"

    if len(items[0]) < world_size:
        return items

    ret = [[] for _ in range(len(items))]

    items_per_rank = len(items[0]) // world_size

    start_idx = rank * items_per_rank
    if rank == world_size - 1:
        end_idx = len(items[0])
    else:
        end_idx = (rank + 1) * items_per_rank

    for i in range(len(items)):
        ret[i] = items[i][start_idx:end_idx]

    return ret
