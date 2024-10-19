#!/usr/bin/env python
import torch
from typing import Dict, Tuple, List

import warnings

from ..utils import get_feature_dir, load_embedding, get_feature_path
from ..utils.how2qa import read_csv
from .common import DejavuDataset
from ..utils.dataset import count_available_features

# Build dataset from the frames
class How2qaDataset(DejavuDataset):
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
            use_start_end=False,
        ):
        super().__init__(
            'how2qa',
            split,
            base_model_name,
            fps,
            return_pixel_values=return_pixel_values,
            return_input_values=return_input_values,
            return_hidden_states=return_hidden_states,
            return_output_states=return_output_states,
            return_compressed=return_compressed,
            reuse_dataset_info=reuse_dataset_info,
            use_coded_order=False,
            use_start_end=use_start_end,
        )

        if self.id_idx is None:
            if use_coded_order:
                raise NotImplementedError("Coded order is not supported for How2QA dataset yet")

            print(f"Creating new info file: {self.dataset_info_path}")
            df = read_csv(split)

            # Count the number of features per video
            num_per_video = {}

            if use_start_end:
                youtube_ids = df['id']
                starts = df['start'].apply(lambda x: int(x * fps))
                ends = df['end'].apply(lambda x: int(x * fps))
            else:
                youtube_ids = df['id'].unique()
                starts = None
                ends = None

            count_info = count_available_features(
                feature_dir=self.feature_dir,
                video_ids=youtube_ids,
                starts=starts,
                ends=ends,
                check_compressed=return_compressed,
            )

            self.id_idx = []    # youtube_id, frameidx
            for youtube_id, start, end, frame_cnt in count_info:
                if frame_cnt == 0:
                    warnings.warn(f'Video {youtube_id} has no features, it probably means error from sanitization')
                    continue

                for i in range(start, end):
                    self.id_idx.append((youtube_id, i))

            self.save_dataset_info()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test How2QA dataset')
    parser.add_argument('SPLIT', choices=['train', 'test', 'frozenbilm'], help='Split to test')
    parser.add_argument('FPS', type=float)
    parser.add_argument('--regenerate-dataset-info', action='store_true', help='Regenerate dataset info')
    parser.add_argument('--use-start-end', action='store_true', help='Use start and end time')
    parser.add_argument('--return-compressed', action='store_true', help='Return compressed features')
    args = parser.parse_args()

    TEST_STEP = 500

    dataset = How2qaDataset(
        split=args.SPLIT,
        base_model_name='openai/clip-vit-large-patch14',
        fps=args.FPS,
        return_output_states=False,
        return_compressed=args.return_compressed,
        reuse_dataset_info=not args.regenerate_dataset_info,
        use_start_end=args.use_start_end,
    )
    for i in range(0, len(dataset), TEST_STEP):
        _ = dataset[i]

    print(f'len: {len(dataset)}')
