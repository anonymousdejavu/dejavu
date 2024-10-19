import torch
from pathlib import Path
from joblib import dump, load
from ..utils import get_feature_dir, get_feature_path, load_embedding

class DejavuDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset,
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
            use_feature_path_v2=False,
            dir_key='feature_dir',
        ):
        self.feature_dir = get_feature_dir(dataset, base_model_name, fps, split, dir_key=dir_key)

        self.return_pixel_values = return_pixel_values
        self.return_input_values = return_input_values
        self.return_hidden_states = return_hidden_states
        self.return_output_states = return_output_states
        self.return_compressed = return_compressed
        self.use_feature_path_v2 = use_feature_path_v2

        # Cached entries for the dataset
        self.dataset_info_path = self.feature_dir / f'dataset_info.joblib'
        if use_coded_order:
            self.dataset_info_path = self.dataset_info_path.with_name(self.dataset_info_path.stem + '_coded_order' + self.dataset_info_path.suffix)
        if use_start_end:
            self.dataset_info_path = self.dataset_info_path.with_name(self.dataset_info_path.stem + '_start_end' + self.dataset_info_path.suffix)

        if reuse_dataset_info and self.dataset_info_path.exists():
            print(f"Reusing existing info file: {self.dataset_info_path}")
            self.id_idx = load(self.dataset_info_path)
        else:
            print("Child class should populate self.id_idx and call save_dataset_info()")
            self.id_idx = None

    def save_dataset_info(self):
        self.dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        dump(self.id_idx, self.dataset_info_path)

    def __len__(self):
        return len(self.id_idx)

    def __getitem__(self, idx):
        # Find the video index
        youtube_id, frame_idx = self.id_idx[idx]

        if not isinstance(frame_idx, torch.Tensor):
            frame_idx = torch.tensor(frame_idx)

        ret = (frame_idx,)
        if self.return_pixel_values:
            p_path = get_feature_path(self.feature_dir, youtube_id, 'p', frame_idx, use_v2=self.use_feature_path_v2)
            ret += (load_embedding(p_path),)
        if self.return_input_values:
            i_path = get_feature_path(self.feature_dir, youtube_id, 'i', frame_idx, use_v2=self.use_feature_path_v2)
            ret += (load_embedding(i_path),)
        if self.return_hidden_states:
            h_path = get_feature_path(self.feature_dir, youtube_id, 'h', frame_idx, use_v2=self.use_feature_path_v2)
            ret += (load_embedding(h_path),)
        if self.return_output_states:
            o_path = get_feature_path(self.feature_dir, youtube_id, 'o', frame_idx, use_v2=self.use_feature_path_v2)
            ret += (load_embedding(o_path),)
        if self.return_compressed:
            T = -3
            c_path = get_feature_path(self.feature_dir, youtube_id, 'c', frame_idx, use_v2=self.use_feature_path_v2)
            cur_embedding = load_embedding(c_path)

            prev_embeddings = []
            for t in range(T, 0):
                if idx + t < 0:
                    prev_embeddings.append(torch.zeros_like(cur_embedding))
                    continue

                prev_id, prev_frame_idx = self.id_idx[idx + t]

                if prev_id != youtube_id:
                    prev_embeddings.append(torch.zeros_like(cur_embedding))
                else:
                    prev_c_path = get_feature_path(self.feature_dir, prev_id, 'c', prev_frame_idx, use_v2=self.use_feature_path_v2)
                    prev_embeddings.append(load_embedding(prev_c_path))

            prev_embeddings.append(cur_embedding)
            ret += (torch.stack(prev_embeddings, dim=1),)

        return ret

    def is_start_of_new_video(self, idx):
        _, frame_idx = self.id_idx[idx]
        return frame_idx == 0