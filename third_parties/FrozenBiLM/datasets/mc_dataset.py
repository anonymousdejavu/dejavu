import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math
from pathlib import Path
import numpy as np
import os

import torch
import time

import sys
if '/workspace' not in sys.path:
    sys.path.insert(0, '/workspace')
from dejavu.utils import add_noise_for_similarity, add_noise_for_mse
from dejavu.utils import load_embedding


class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path,
        max_feats=10,
        features_dim=768,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
        wrong_qids_csv_path=None,
        fps=1,
    ):
        self.data = pd.read_csv(csv_path)
        if wrong_qids_csv_path is not None:
            self.data = pd.read_csv(wrong_qids_csv_path)
        self.data.fillna({'a0': '', 'a1': '', 'a2': '', 'a3': '', 'a4': ''}, inplace=True)

        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        # self.features_dir = Path('/mnt/ssd2/dataset/how2qa/openai_clip-vit-large-patch14') 
        # self.features_dir = Path('/mnt/ssd2/dataset/how2qa/openai_clip-vit-large-patch14') 
        self.features_dir = Path(features_path) 
        self.features_dim = features_dim
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        mc = 0
        while f"a{mc}" in self.data:
            mc += 1
        self.mc = mc
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix
        self.fps = fps
        self.max_feats = max_feats

    def __len__(self):
        return len(self.data)

    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, video_id, start, end):
        if start is not None and not math.isnan(start):
            start = int(int(start) * self.fps)
            end = int(int(end) * self.fps) 
        else:
            raise NotImplementedError

        frame_features = []
        video_id = '_'.join(video_id.split('_')[:-2])
        # We already applied it during sanitization
        # time_base = int(video_id.split('_')[-2])
        # start += time_base
        # end += time_base
        for i in range(start, end):
            feature_path = self.features_dir / f'{video_id}_o_{i}.npz'
            try:
                features = load_embedding(feature_path, return_pt=True)
                # For noise injection
                noise_level = os.environ.get('INJECT_NOISE', None)
                if noise_level is not None:
                    noise_level = float(noise_level)
                    if os.environ.get('INJECT_NOISE_MSE', None) is not None:
                        features = add_noise_for_mse(features, noise_level)
                    else:
                        features = add_noise_for_similarity(features, noise_level)
                frame_features.append(features)
            except Exception as e:
                print(f'Feature {feature_path} not loaded, needed ({start}, {end}): {e}')
                break

        if len(frame_features) > self.max_feats:
            # sample frames
            # frame_idxs = np.linspace(0, len(frame_features) - 1, self.max_feats)
            # frame_idxs = np.round(frame_idxs).astype(int)
            # frame_features = [frame_features[i] for i in frame_idxs]
            step = len(frame_features) // self.max_feats
            tmp = []
            for j in range(self.max_feats):
                tmp.append(frame_features[j * step])
            frame_features = tmp

        video_len = len(frame_features)
        for i in range(video_len, self.max_feats):
            # Fill the rest with zeros
            frame_features.append(th.zeros(self.features_dim))
        video = th.stack(frame_features)

        return video, video_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]

        # get start, end
        start = self.data["start"].values[idx]
        end = self.data["end"].values[idx]

        time_base = int(video_id.split('_')[-2])
        start_old = start - time_base
        end_old = end - time_base

        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get subs
        if self.subs:
            subs = self._get_subtitles(video_id, start_old, end_old)
        else:
            subs = ""

        # get features
        video, video_len = self._get_video(video_id, start, end)

        # get answer id
        answer_id = -1  # for hidden set testing
        if "answer_id" in self.data:
            answer_id = self.data["answer_id"].values[idx]

        text = []
        for i in range(self.mc):
            ai = self.data[f"a{i}"].values[idx].capitalize().strip()
            text.append(self._get_text(subs, ai, self.mask, question))

        qid = idx
        if "qid" in self.data:
            qid = int(self.data["qid"].values[idx])
        
        return {
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": qid,
            "answer_id": answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "how2qa":
        if split == "train":
            csv_path = args.how2qa_train_csv_path
        elif split == "val":
            csv_path = args.how2qa_val_csv_path
        elif split == "test":
            csv_path = args.how2qa_val_csv_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.how2qa_subtitles_path
        features_path = args.how2qa_features_path
    elif dataset_name == "tvqa":
        if split == "train":
            csv_path = args.tvqa_train_csv_path
        elif split == "val":
            csv_path = args.tvqa_val_csv_path
        elif split == "test":
            csv_path = args.tvqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.tvqa_subtitles_path
        features_path = args.tvqa_features_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path=features_path,
        max_feats=args.max_feats,
        features_dim=args.features_dim,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
        wrong_qids_csv_path=args.wrong_qids_csv_path,
        fps=args.fps,
    )
