from . import How2qaDataset, MsrvttDataset

import torch

class StackFrameDatasetMixin:
    def __init__(self, pattern):
        # ex) [0, 2, 4, 6, 5] would return
        # 1. frame 0, 2, 4, 6, 5
        # 2. frame step, step+2, step+4, step+6, step+5
        # When there is not enough frames left to make pattern, fill with last frame

        assert pattern[0] == 0, 'The first element of stack_pattern should be 0'

        self.pattern = pattern
        self.num_frames = len(pattern)
        if step is None:
            step = self.num_frames
        self.step = step

        self.super_idx = []

        self.len = 0

        pattern_max = max(pattern)

        idx = 0
        while True:
            if idx + pattern_max >= len(self.id_idx):
                break
            start_youtube_id, _ = self.id_idx[idx]
            for p in pattern[1:]:
                cur_idx = idx + p
                cur_youtube_id, _ = self.id_idx[cur_idx]
                if start_youtube_id != cur_youtube_id:
                    # Skip if there is not enough frames left to make pattern
                    idx = cur_idx
                    break

            if start_youtube_id != cur_youtube_id:
                continue

            self.super_idx.append(idx)
            idx += self.step


    def __getitem__(self, idx):
        super_idx = self.super_idx[idx]
        rets = []
        for p in self.pattern:
            cur_idx = super_idx + p
            rets.append(super().__getitem__(cur_idx))

        if len(rets) == 1:
            rets = rets[0]
        else:
            rets = list(zip(*rets))
            rets = [torch.stack(r, dim=0) for r in rets]

        return rets

    def __len__(self):
        return len(self.super_idx)


class StackedHow2qaDataset(StackFrameDatasetMixin, How2qaDataset):
    def __init__(
            self,
            pattern,
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
            step=None
    ):
        How2qaDataset.__init__(
            self, 
            split,
            base_model_name,
            fps,
            return_pixel_values=return_pixel_values,
            return_input_values=return_input_values,
            return_hidden_states=return_hidden_states,
            return_output_states=return_output_states,
            return_compressed=return_compressed,
            reuse_dataset_info=reuse_dataset_info,
            use_coded_order=use_coded_order
        )
        StackFrameDatasetMixin.__init__(self, pattern, step=step)


class StackedMsrvttDataset(StackFrameDatasetMixin, MsrvttDataset):
    def __init__(
            self,
            pattern,
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
            step=None
    ):
        MsrvttDataset.__init__(
            self, 
            split,
            base_model_name,
            fps,
            return_pixel_values=return_pixel_values,
            return_input_values=return_input_values,
            return_hidden_states=return_hidden_states,
            return_output_states=return_output_states,
            return_compressed=return_compressed,
            reuse_dataset_info=reuse_dataset_info,
            use_coded_order=use_coded_order
        )
        StackFrameDatasetMixin.__init__(self, pattern, step=step)

