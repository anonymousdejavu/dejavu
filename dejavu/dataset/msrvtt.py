#!/usr/bin/env python
import torch
import torch.nn as nn

from torch.utils.data import Dataset

import numpy as np

import warnings

from joblib import dump, load
from typing import Dict, Tuple, List, Union

from ..utils import PREDEFINED_PATHS, get_sanitized_path, get_coded_order_path
from ..utils.dataset import get_feature_dir, get_feature_dir_malus03, load_embedding
from ..utils.msrvtt import read_csv
from .common import DejavuDataset


# Build dataset from the frames
class MsrvttDataset(DejavuDataset):
    def __init__(
            self, 
            split,
            base_model_name,
            fps,
            return_pixel_values=False,
            return_input_values=True,
            return_hidden_states=False,
            return_output_states=True,
            return_compressed=False,
            reuse_dataset_info=True,
            use_coded_order=False,
        ):
        super().__init__(
            'msrvtt',
            split,
            base_model_name,
            fps,
            return_pixel_values=return_pixel_values,
            return_input_values=return_input_values,
            return_hidden_states=return_hidden_states,
            return_output_states=return_output_states,
            return_compressed=return_compressed,
            reuse_dataset_info=reuse_dataset_info,
            use_coded_order=use_coded_order,
        )

        if self.id_idx is None:
            print(f"Creating new info file: {self.dataset_info_path}")
            sanitized_path = get_sanitized_path('msrvtt', split)
            df = read_csv(sanitized_path)
            youtube_ids = [d for d in df['video_id']]

            def is_feature_path(path):
                return path.is_file() and path.name.endswith('.npz') and '_p_' in path.name

            # Count the number of features per video
            # StartFrame, EndFrame
            num_per_video = {}
            for feature_path in self.feature_dir.iterdir():
                if not is_feature_path(feature_path):
                    continue
                fname_split = feature_path.stem.split('_')
                youtube_id = '_'.join(fname_split[:-2])
                if youtube_id not in youtube_ids:
                    continue
                frame_idx = int(fname_split[-1])

                cnt = num_per_video.get(youtube_id, 0)
                num_per_video[youtube_id] = cnt + 1

            self.id_idx = []
            for youtube_id in youtube_ids:
                video_length = num_per_video.get(youtube_id, 0)
                if video_length == 0:
                    warnings.warn(f'Video {youtube_id} has no features, it probably means error from sanitization')
                    continue

                if use_coded_order:
                    coded_order_path = get_coded_order_path(self.feature_dir, youtube_id)
                    coded_order = load_embedding(coded_order_path)

                    max_coded_order = coded_order.max()
                    if max_coded_order + 1 < video_length:
                        warnings.warn(f"Video {youtube_id} has coded order with max {max_coded_order} < {video_length}")
                    valid_idxs = filter(lambda x: x < video_length, coded_order)
                else:
                    valid_idxs = range(video_length)

                for i in valid_idxs:
                    self.id_idx.append((youtube_id, i))

            self.save_dataset_info()


if __name__ == '__main__':
    TEST_STEP = 500
    REUSE_DATASET_INFO = True
    # Use this to populate dataset info
    # REUSE_DATASET_INFO = False

    dataset = MsrvttDataset(
        split='train',
        base_model_name='openai/clip-vit-base-patch16',
        fps=1,
        reuse_dataset_info=REUSE_DATASET_INFO,
        use_coded_order=True,
    )
    for i in range(0, len(dataset), TEST_STEP):
        _ = dataset[i]

    dataset = MsrvttDataset(
        split='test',
        base_model_name='openai/clip-vit-base-patch16',
        fps=1,
        reuse_dataset_info=REUSE_DATASET_INFO,
        use_coded_order=True,
    )
    for i in range(0, len(dataset), TEST_STEP):
        _ = dataset[i]