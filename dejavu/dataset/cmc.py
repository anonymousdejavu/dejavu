from . import How2qaDataset, MsrvttDataset

import torch

class CmcDatasetMixin:
    def __init__(self, num_frames):
        # 1. Count the number of frames per video
        # FIXME: The following sheme will not work if same_video id is repeated in different segment or frame_idx does not start from 0
        id2start = {}
        id2len = {}
        for idx, (video_id, frame_idx) in enumerate(self.id_idx):
            if frame_idx == 0:
                id2start[video_id] = idx
            # This should overwrite the previous value
            id2len[video_id] = frame_idx + 1

        # 2. Create a list of super indices
        #    The middle frame of the num_frame should come first
        # i) if number of frames is less than pattern, fill with the last frame
        self.cmc_states = []

        for video_id, video_len in id2len.items():
            inner_start_idx = id2start[video_id]
            for batch_start_idx in range(0, video_len, num_frames):
                inner_idxs = [] # Used to index super dataset
                frame_idxs = []
                valid_masks = []

                # The middle frame goes in first
                cur_len = min(video_len - batch_start_idx, num_frames)
                middle_idx = batch_start_idx + cur_len // 2
                inner_idxs.append(inner_start_idx + middle_idx)
                frame_idxs.append(middle_idx)
                valid_masks.append(True)

                for p in range(num_frames):
                    cur_idx = batch_start_idx + p
                    if cur_idx == middle_idx:
                        continue # Middle frame is already added
                    elif cur_idx >= video_len:
                        inner_idxs.append(inner_idxs[-1])
                        frame_idxs.append(frame_idxs[-1])
                        valid_masks.append(False)
                    else:
                        inner_idxs.append(inner_start_idx + cur_idx)
                        frame_idxs.append(cur_idx)
                        valid_masks.append(True)
                    
                self.cmc_states.append((video_id, inner_idxs, frame_idxs, valid_masks))

    def __getitem__(self, idx):
        video_id, inner_idxs, frame_idxs, valid_masks = self.cmc_states[idx]

        rets = []
        for inner_idx in inner_idxs:
            rets.append(super().__getitem__(inner_idx))

        # Stack the frames
        if len(rets) == 1:
            rets = (rets[0], )
        else:
            rets = tuple(zip(*rets))
            rets = tuple(torch.stack(r, dim=0) for r in rets)

        frame_idxs = torch.tensor(frame_idxs)
        valid_masks = torch.tensor(valid_masks)

        assert (frame_idxs == rets[0]).all(), 'Frame idxs mismatch'

        rets = (video_id, valid_masks, *rets)

        return rets

    def __len__(self):
        return len(self.cmc_states)


class How2qaCmcDataset(CmcDatasetMixin, How2qaDataset):
    def __init__(
            self,
            num_frames,
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
        CmcDatasetMixin.__init__(self, num_frames)

class MsrvttCmcDataset(CmcDatasetMixin, MsrvttDataset):
    def __init__(
            self,
            num_frames,
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
        CmcDatasetMixin.__init__(self, num_frames)


