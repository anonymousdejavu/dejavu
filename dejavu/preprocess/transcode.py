import pandas as pd
import ray
from pathlib import Path
import subprocess
import shlex
import argparse
import subprocess
import json
from tqdm import tqdm
import numpy as np

from ..utils import how2qa, msrvtt

from ..utils.dataset import ray_get_with_tqdm
from ..utils import PREDEFINED_PATHS, get_cropped_video_path
from ..utils.preprocess import get_transcode_dir, get_available_datasets, get_available_splits, filter_existing_files, split_per_rank

@ray.remote(num_cpus=4)
def run_transcode(input_path, output_path, log_path, width, height, fps, start_end=None, dry_run=False, overwrite=False):
    CMD = 'ffmpeg'
    if overwrite:
        CMD += ' -y'
    CMD += f" -i '{input_path}'"
    if start_end is not None:
        CMD += f' -ss {start_end[0]} -to {start_end[1]}'
    CMD += f' -vf "fps={fps},crop=\'min(iw,ih):min(iw,ih)\',scale={width}:{height}" -b_strategy 0 -bf 3'
    CMD += f' -c:v libx264 -preset slow -crf 22'
    CMD += f' -x264opts force-cfr:no-scenecut:subme=0:me=dia:ref=2'
    CMD += f' -threads 4'
    CMD += f' -an'
    CMD += f" '{output_path}'"

    stdout_path = log_path.with_suffix('.stdout')
    stderr_path = log_path.with_suffix('.stderr')

    print(CMD)
    if dry_run:
        return
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        handle = subprocess.Popen(
                shlex.split(CMD),
                stdout=stdout,
                stderr=stderr
                )
        handle.communicate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
       "DATASET",
       choices=get_available_datasets(),
    )
    parser.add_argument(
        "SPLIT",
        choices=get_available_splits(exclude_sanitized=True),
    )
    parser.add_argument('FPS', type=str)
    parser.add_argument("--dry-run", action='store_true', help='Dry run')
    parser.add_argument("--overwrite", action='store_true', help='Overwrite existing files')
    parser.add_argument("--override-csv-path", type=str, default=None)
    parser.add_argument("--base-model-name", type=str, default=None, choices=['openai/clip-vit-base-patch16', 'openai/clip-vit-large-patch14'])

    # For SLURM
    parser.add_argument('--rank', type=int, default=None, help='Rank of the node (for SLURM)')
    parser.add_argument('--world-size', type=int, default=None, help='Total number of nodes (for SLURM)')
    args = parser.parse_args()

    if args.override_csv_path is not None:
        assert args.DATASET == 'how2qa', 'Override csv path is only supported for how2qa'

    start_ends = None
    if args.DATASET == 'how2qa':
        if args.override_csv_path is None:
            split_or_path = args.SPLIT
            return_time = False
        else:
            split_or_path = args.override_csv_path
            return_time = True
        youtube_ids, video_paths, start_ends = how2qa.get_youtube_ids_and_paths(
            split_or_path=split_or_path,
            fps=None,
            return_time=return_time
        )
    elif args.DATASET == 'msrvtt':
        youtube_ids, video_paths, _ = msrvtt.get_video_ids_and_paths(split=args.SPLIT, fps=None)
        if args.base_model_name is None:
            print(f'Base model name is not provided. Assuming openai/clip-vit-base-patch16')
            width, height = 224, 224
    else:
        raise NotImplementedError
        
    if args.base_model_name is not None:
        if args.base_model_name == 'openai/clip-vit-large-patch14':
            width, height = 256, 256
        elif args.base_model_name == 'openai/clip-vit-base-patch16':
            width, height = 224, 224
        else:
            raise NotImplementedError(f'Base model name {args.base_model_name} is not supported')

    output_paths = []
    for idx, video_path in enumerate(video_paths):
        output_dir = get_transcode_dir(args.DATASET, args.FPS, args.base_model_name)
        output_path = output_dir / video_path.with_suffix('.mp4').name
        if start_ends is not None:
            start_end = start_ends[idx]
            output_path = get_cropped_video_path(output_path, start_end[0], start_end[1])
        output_paths.append(output_path)

    if not args.overwrite:
        output_paths, video_paths = filter_existing_files(output_paths, video_paths)

    tmp_output = []
    tmp_video = []
    for output_path, video_path in zip(output_paths, video_paths):
        if output_path in tmp_output:
            continue
        tmp_output.append(output_path)
        tmp_video.append(video_path)
    output_paths = tmp_output
    video_paths = tmp_video

    if args.world_size is not None:
        if args.rank is None:
            raise ValueError('Rank must be provided')
        world_size = args.world_size
        rank = args.rank

        if rank >= world_size:
            raise ValueError('Rank must be less than world size')

        output_paths, video_paths = split_per_rank(rank, world_size, output_paths, video_paths)

    ret = []

    for idx in range(len(video_paths)):
        video_path = video_paths[idx]
        output_path = output_paths[idx]
        if start_ends is not None:
            start_end = start_ends[idx]
        else:
            start_end = None

        # Create output directory
        output_dir = Path(output_path).parent
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)


        log_path = log_dir / video_path.with_suffix('.log').name

        r = run_transcode.remote(
            video_path,
            output_path,
            log_path,
            width,
            height,
            args.FPS,
            start_end=start_end,
            dry_run=args.dry_run,
            overwrite=args.overwrite
        )
        ret.append(r)

    ray_get_with_tqdm(ret)