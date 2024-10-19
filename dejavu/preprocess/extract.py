#!/usr/bin/env python

import argparse
from pathlib import Path

import ray
from ray.util import ActorPool
import torch
from transformers import CLIPProcessor
import numpy as np
import os
from tqdm import tqdm

from ..utils import how2qa, msrvtt
from ..utils import PREDEFINED_PATHS

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

from ..utils.dataset import get_all_frames, get_feature_dir, load_embedding, save_embedding # sample_16_frames
from ..utils.preprocess import get_available_datasets, get_available_splits, get_feature_path, split_per_rank
from ..model.clip import CLIPVisionModelWithProjection

class VideoProcessor:
    def __init__(
            self,
            feature_dir,
            processor_factory,
            model_factory,
            batch_size,
            device,
            dry_run=False,
            use_feature_v2=False,
            target_features=['i', 'p', 'o', 'h'],
        ):
        self.feature_dir = Path(feature_dir)
        self.processor = None
        self.model = None
        self.device = torch.device(device)
        if device == 'cuda':
            gpu_id = ray.get_gpu_ids()[0]
            print(f'Using GPU {gpu_id}')
        self.batch_size = batch_size
        self.processor_factory = processor_factory
        self.model_factory = model_factory
        self.dry_run = dry_run
        self.use_feature_v2 = use_feature_v2
        self.target_features = target_features

    def get_processor(self):
        if self.processor is None:
            self.processor = self.processor_factory()
        return self.processor

    def get_model(self):
        if self.model is None:
            model= self.model_factory()
            self.model = model.to(self.device)
        return self.model

    def save_embedding(self, embedding, path):
        if self.dry_run:
            print(f'DRYRUN: {path}')
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        save_embedding(embedding, path)

    def process_video(
            self,
            video_id,
            video_path,
            overwrite=False,
            start=None,
            end=None,
        ):
        if not Path(video_path).exists():
            print(f'Video does not exist: {video_path}')
            return

        frames = get_all_frames(video_path)

        if not frames:
            print('No frames available')
            return

        if start is None:
            start = 0
        if end is None:
            end = start + len(frames)

        if not overwrite:
            all_features_avaiable = self.check_all_features_available(video_id, start, end, self.target_features)
            if all_features_avaiable:
                print(f'All frames already processed for {video_id}')
                return

        # frames = frames[start:end]
        print(f'Processing {video_id} with {len(frames)} frames')

        processor = self.get_processor()

        pixel_values = processor.image_processor(
            images=frames, return_tensors="pt", padding=True
        )["pixel_values"]

        if self.batch_size == -1:
            batch_size = len(pixel_values)
        else:
            batch_size = self.batch_size
            # At first, batch size should be 5 (including 0th frame)

        outer_batch_idx = 0
        while outer_batch_idx < len(pixel_values):
            batched_frames = frames[outer_batch_idx:outer_batch_idx+batch_size]
            batch_pixel_values = pixel_values[outer_batch_idx:outer_batch_idx+batch_size]
            batch_pixel_values_on_device = batch_pixel_values.to(self.device)

            if not self.target_features == ['i']:
                model = self.get_model()

                with torch.no_grad():
                    outputs = model(
                        pixel_values=batch_pixel_values_on_device,
                        output_hidden_states=True,
                    )
                    image_embeds = outputs.image_embeds.cpu()
                    hidden_states = outputs.hidden_states[-1].cpu()
                    second_hidden_states = outputs.hidden_states[-2].cpu()

            for inner_batch_idx in range(len(batch_pixel_values)):
                frame_idx = start + outer_batch_idx + inner_batch_idx

                i = batch_pixel_values[inner_batch_idx]
                i_path = get_feature_path(self.feature_dir, video_id, 'i', frame_idx, use_v2=self.use_feature_v2)

                if 'i' in self.target_features:
                    self.save_embedding(i, i_path)
                
                if 'p' in self.target_features:
                    p = batched_frames[inner_batch_idx]
                    p_path = get_feature_path(self.feature_dir, video_id, 'p', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(p, p_path)

                if 'o' in self.target_features:
                    o = image_embeds[inner_batch_idx]
                    o_path = get_feature_path(self.feature_dir, video_id, 'o', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(o, o_path)

                if 'h' in self.target_features:
                    h = hidden_states[inner_batch_idx]
                    h_path = get_feature_path(self.feature_dir, video_id, 'h', frame_idx, use_v2=self.use_feature_v2)
                    self.save_embedding(h, h_path)   

                if 'hh' in self.target_features:
                    hh = second_hidden_states[inner_batch_idx]
                    hh_path = get_feature_path(self.feature_dir, video_id, 'hh', frame_idx, use_v2=self.use_feature_v2)
                    hh_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_embedding(hh, hh_path)

            outer_batch_idx += batch_size

    def check_all_features_available(
        self,
        video_id,
        start,
        end,
        feature_types=['p', 'i', 'o', 'h'],
    ):
        all_frames_loaded = True
        try:
            for frame_idx in range(start, end):
                # try to load from file
                for feature_type in feature_types:
                    feature_path = get_feature_path(self.feature_dir, video_id, feature_type, frame_idx, use_v2=self.use_feature_v2)
                    load_embedding(feature_path)

        except Exception as e:
            print(f'Error: {e}')
            all_frames_loaded = False

        return all_frames_loaded


def extract_features(
        youtube_ids,
        video_paths,
        feature_dir,
        processor_factory,
        model_factory,
        args,
        device,
        use_feature_v2,
        starts=None,
        ends=None,
        overwrite=False,
        target_features=['i', 'p', 'o', 'h'],
    ):
    num_workers = args.num_workers
    num_gpus = args.num_gpus
    batch_size = args.batch_size
    dry_run = args.dry_run

    def maybe_remote(device):
        if device == 'cpu':
            return ray.remote
        elif device == 'cuda':
            gpu_per_worker = num_gpus / num_workers
            return ray.remote(num_gpus=gpu_per_worker)
        else:
            raise NotImplementedError(f'Unknown device: {device}')

    DecoratedVideoProcessor = maybe_remote(device)(VideoProcessor)

    workers = [DecoratedVideoProcessor.remote(
        feature_dir,
        processor_factory,
        model_factory,
        batch_size,
        device,
        dry_run,
        use_feature_v2,
        target_features,
    ) for _ in range(num_workers)]
    pool = ActorPool(workers)

    ret = []

    def submit_to_actor(actor, idx):
        return actor.process_video.remote(
            youtube_ids[idx],
            video_paths[idx],
            overwrite=overwrite,
            start=None if starts is None else starts[idx],
            end=None if ends is None else ends[idx],
        )

    ret = pool.map_unordered(submit_to_actor, range(len(youtube_ids)))

    if not dry_run:
        Path(feature_dir).mkdir(parents=True, exist_ok=True)

    _ = [a for a in tqdm(ret, total=len(youtube_ids))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("DATASET", choices=get_available_datasets())
    parser.add_argument("SPLIT", choices=get_available_splits())
    parser.add_argument('FPS', type=float)
    parser.add_argument(
        'BASE_MODEL_NAME',
        type=str,
        choices=[
            'openai/clip-vit-base-patch16',
            'openai/clip-vit-large-patch14'
        ]
    )
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs to use, (0 for CPU)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help='Do not save features')
    parser.add_argument("--override-csv-path", type=str, default=None, help='Override CSV path')
    parser.add_argument("--override-csv-path-video-not-cropped", action='store_true', help='Videos for overwritten CSV is not cropped')
    parser.add_argument("--use-compile", action='store_true', help='Use compiled model')
    parser.add_argument("--use-mixed-precision", action='store_true', help='Use mixed precision')
    parser.add_argument("--overwrite", action='store_true', help='Overwrite existing features')
    parser.add_argument("--extract-hh", action='store_true', help='Extract second to last hidden states')
    parser.add_argument("--extract-i-only", action='store_true', help='Extract input features only')
    parser.add_argument("--extract-msrvtt-feature", action='store_true', help='Extract input features only')
    parser.add_argument("--target-features", type=str, default='i,p,o,h', help='Target features to extract (comma separated)')

    # SLURM
    parser.add_argument('--rank', type=int, default=None, help='Rank of the node (for SLURM)')
    parser.add_argument('--world-size', type=int, default=None, help='Total number of nodes (for SLURM)')

    args = parser.parse_args()

    target_features = args.target_features.split(',')
    print(f'Extracting features: {target_features}')

    if args.use_mixed_precision:
        torch.set_float32_matmul_precision('high')

    if args.override_csv_path is not None:
        assert args.DATASET == 'how2qa', 'Override csv path is only supported for how2qa'

    feature_dir = get_feature_dir(
        args.DATASET,
        args.BASE_MODEL_NAME,
        args.FPS,
        args.SPLIT,
        dir_key='msrvtt_feature_dir' if args.extract_msrvtt_feature else 'feature_dir'
    )

    if args.DATASET == 'msrvtt' or args.extract_msrvtt_feature:
        name_or_checkpoint = PREDEFINED_PATHS['msrvtt']['CLIP4Clip_checkpoint']
        print(f'Extracting MSRVTT features from {name_or_checkpoint}')
    else:
        name_or_checkpoint = args.BASE_MODEL_NAME

    def processor_factory(*_):
        processor = CLIPProcessor.from_pretrained(
            args.BASE_MODEL_NAME,
            cache_dir=CACHE_DIR
        )
        return processor

    def model_factory(*_):
        model = CLIPVisionModelWithProjection.from_pretrained(
            name_or_checkpoint,
            cache_dir=CACHE_DIR
        )
        if args.use_mixed_precision:
            torch.set_float32_matmul_precision('high')
        if args.use_compile:
            model.compile()
        return model

    use_feature_v2 = False
    if args.DATASET == 'how2qa':
        if args.override_csv_path is None:
            split_or_path = args.SPLIT
        else:
            split_or_path = args.override_csv_path

        youtube_ids, video_paths, start_ends = how2qa.get_youtube_ids_and_paths(
            split_or_path=split_or_path,
            fps=args.FPS,
            return_time=False
        )
        # if start_ends is not None:
        #     if args.override_csv_path_video_not_cropped:
        #         starts = [int(s * args.FPS) for s, _ in start_ends]
        #         ends = [int(e * args.FPS) for _, e in start_ends]
        #     else:
        # else:
        # starts = None
        # ends = None
        # video_paths = [v.parent / f'{v.stem}_{s}_{e}.mp4' for v, (s, e) in zip(video_paths, start_ends)]
        # starts = [int(s * args.FPS) for s, _ in start_ends]
        starts = None
        ends = None
        # starts = [int(s * args.FPS) for s, _ in start_ends]
        # ends = [int(e * args.FPS) for _, e in start_ends]

    elif args.DATASET == 'msrvtt':
        youtube_ids, video_paths, _ = msrvtt.get_video_ids_and_paths(split=args.SPLIT, fps=args.FPS)
        starts = None
        ends = None
    else:
        raise NotImplementedError

    if args.world_size is not None:
        if args.rank is None:
            raise ValueError('Rank must be provided')
        world_size = args.world_size
        rank = args.rank

        if rank >= world_size:
            raise ValueError('Rank must be less than world size')

        youtube_ids, video_paths = split_per_rank(rank, world_size, youtube_ids, video_paths)


    if args.num_gpus > 0:
        device = 'cuda'
        ray.init(num_gpus=args.num_gpus)
    else:
        device = 'cpu'
        ray.init()

    extract_features(
        youtube_ids,
        video_paths,
        feature_dir,
        processor_factory,
        model_factory,
        args,
        device=device,
        use_feature_v2=use_feature_v2,
        starts=starts,
        ends=ends,
        overwrite=args.overwrite,
        target_features=target_features,
    )
