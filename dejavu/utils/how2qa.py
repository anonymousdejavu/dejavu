import pandas as pd
from pathlib import Path
import numpy as np

from ..utils import PREDEFINED_PATHS, load_embedding, get_cropped_video_path
from ..utils.preprocess import get_transcode_dir

def read_csv(split_or_path):
    if split_or_path in ['train', 'test']:
        df = pd.read_csv(
            PREDEFINED_PATHS['how2qa']['split'][split_or_path],
            names=['id', 'interval', 'wrong1', 'wrong2', 'wrong3', 'question', 'correct', 'query_id' ]
        )
        start_times = []
        end_times = []
        for interval in df['interval']:
            start, end = interval.strip('[]').split(':')
            start_times.append(float(start))
            end_times.append(float(end))
        df['start'] = start_times
        df['end'] = end_times
    elif split_or_path == 'frozenbilm':
        df = pd.read_csv(PREDEFINED_PATHS['how2qa']['split'][split_or_path])
        youtube_ids = []
        base_times = []
        for video_id in df['video_id']:
            splitted = video_id.split('_')
            youtube_ids.append('_'.join(splitted[:-2]))
            base_times.append(int(splitted[-2]))
        df['id'] = youtube_ids
        # FrozenBiLM has special way of formatting start and end time
        df['start'] += base_times
        df['end'] += base_times
    else:
        df = pd.read_csv(split_or_path)
    return df

def get_video_paths(youtube_ids, fps):
    if fps is None:
        video_dir = Path(PREDEFINED_PATHS['how2qa']['video'])
    else:
        video_dir = get_transcode_dir('how2qa', fps)
    video_paths = [video_dir / f'{youtube_id}.mp4' for youtube_id in youtube_ids]
    return video_paths


def get_youtube_ids_and_paths(split_or_path, fps, return_time=False):
    df = read_csv(split_or_path)
    if not return_time:
        youtube_ids = df['id'].unique()
        start_ends = None
        video_paths = get_video_paths(youtube_ids, fps)
    else:
        id_start_end_list = list(set(
            (row['id'], row['start'], row['end'])
            for _, row in df.iterrows()
        ))
        youtube_ids, starts, ends = zip(*id_start_end_list)
        starts = [int(start) for start in starts]
        ends = [int(end) for end in ends]
        video_paths = get_video_paths(youtube_ids, fps)
        start_ends = list(zip(starts, ends))

        assert len(video_paths) == len(start_ends)

        if fps is not None:
            for idx, (video_path, start_end) in enumerate(zip(video_paths, start_ends)):
                start, end = start_end
                video_path = get_cropped_video_path(video_path, start, end)
                video_paths[idx] = video_path
    
    return youtube_ids, video_paths, start_ends

def is_valid_row(feature_dir, row):
    youtube_id = row['id']
    start = int(row['start'])
    end = int(row['end'])

    valid = True
    try:
        for frame_idx in range(start, end):
            feature_path = feature_dir / f'{youtube_id}_o_{frame_idx}.npz'
            load_embedding(feature_path)
    except Exception as e:
        print(f"While looking for {youtube_id}({start}~{end}): {e}")
        valid = False

    return valid