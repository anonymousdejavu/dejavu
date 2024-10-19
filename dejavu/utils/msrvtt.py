import pandas as pd
from pathlib import Path

from ..utils.dataset import load_embedding
from ..utils import PREDEFINED_PATHS
from ..utils.preprocess import get_transcode_dir

def read_csv(split_or_path):
    if split_or_path in ['train', 'test']:
        df = pd.read_csv(PREDEFINED_PATHS['msrvtt']['split'][split_or_path])
    else:
        df = pd.read_csv(split_or_path)

    return df

def is_valid_row(feature_dir, row):
    video_id = row['video_id']
    valid = True
    try:
        feature_path = feature_dir / f'{video_id}_o_0.npz'
        load_embedding(feature_path)
    except Exception as e:
        print(f"While looking for {video_id}: {e}")
        valid = False

    return valid

def get_video_paths(video_ids, fps):
    if fps is None:
        video_dir = Path(PREDEFINED_PATHS['msrvtt']['video'])
    else:
        video_dir = get_transcode_dir('msrvtt', fps)
    video_paths = [video_dir / f'{video_id}.mp4' for video_id in video_ids]
    return video_paths


def get_video_ids_and_paths(split, fps):
    video_ids = read_csv(split)['video_id']
    video_paths = get_video_paths(video_ids, fps)
    start_ends = None
    
    return video_ids, video_paths, start_ends
