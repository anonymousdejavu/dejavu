#!/usr/bin/env python
import torch
from typing import Dict, Tuple, List, Union
import numpy as np

from . import How2qaDataset, MsrvttDataset

class EvaluationDatasetMixin:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        id2start = {}
        id2len = {}
        for idx, (video_id, frame_idx) in enumerate(self.id_idx):
            if frame_idx == 0:
                id2start[video_id] = idx
            # This should overwrite the previous value
            id2len[video_id] = frame_idx + 1


        batches = [[] for _ in range(self.batch_size)]
        per_batch_len = np.zeros(self.batch_size, dtype=int)

        for video_id, video_len in id2len.items():
            valid_video_len = (video_len // 4) * 4
            if valid_video_len > 0:
                # greedy fill in per batch
                min_idx = np.argmin(per_batch_len)
                start_idx = id2start[video_id]
                for i in range(0, valid_video_len, 4):
                    batches[min_idx].append((video_id, start_idx + i))
                per_batch_len[min_idx] += valid_video_len
        
        self.super_idx = list(zip(*batches))

        self.pattern = [3, 1, 0, 2]

    def __len__(self):
        return len(self.super_idx)
    
    def __getitem__(self, idx: int):
        ret = []

        for video_id, start_idx in self.super_idx[idx]:
            stack = []
            for i in self.pattern:
                items = super().__getitem__(start_idx + i)
                stack.append(items)
            # batch_size items stacked to the shape of [frame, ...]
            stacked_items = [torch.stack(r, dim=0) for r in zip(*stack)]
            ret.append(stacked_items)
        
        # items stacked to the shape of [frame, batch_size, ...]
        ret = [torch.stack(r, dim=1) for r in zip(*ret)]

        return ret




class How2qaEvaluationDataset(EvaluationDatasetMixin, How2qaDataset):
    def __init__(
            self,
            batch_size,
            split,
            base_model_name,
            fps,
            return_pixel_values=False,
            return_input_values=True,
            return_hidden_states=False,
            return_output_states=True,
            return_compressed=True,
            reuse_dataset_info=True,
    ):
        assert return_compressed, "EvaluationDataset requires return_compressed=True"
        How2qaDataset.__init__(
            self,
            split,
            base_model_name,
            fps,
            return_pixel_values,
            return_input_values,
            return_hidden_states,
            return_output_states,
            return_compressed,
            reuse_dataset_info
        )
        EvaluationDatasetMixin.__init__(self, batch_size)

class MsrvttEvaluationDataset(EvaluationDatasetMixin, MsrvttDataset):
    def __init__(
            self,
            batch_size,
            split,
            base_model_name,
            fps,
            return_pixel_values=False,
            return_input_values=True,
            return_hidden_states=False,
            return_output_states=True,
            return_compressed=True,
            reuse_dataset_info=True,
    ):
        assert return_compressed, "EvaluationDataset requires return_compressed=True"
        MsrvttDataset.__init__(
            self,
            split,
            base_model_name,
            fps,
            return_pixel_values,
            return_input_values,
            return_hidden_states,
            return_output_states,
            return_compressed,
            reuse_dataset_info
        )
        EvaluationDatasetMixin.__init__(self, batch_size)


if __name__ == '__main__':
    dataset = MsrvttEvaluationDataset(
        batch_size=16,
        split='test',
        base_model_name='openai/clip-vit-base-patch16',
        fps=1,
    )

    print("len(dataset):", len(dataset))

    items = dataset[0]
    for idx, item in enumerate(items):
        print(f"item[{idx}].shape:", item.shape)