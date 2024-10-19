import ray
from pathlib import Path
import subprocess
import shlex
import argparse
import subprocess
# from ray.experimental.tqdm_ray import tqdm as tqdm_ray

import cv2

from dejavu.utils import get_available_datasets, get_available_splits, get_feature_dir, get_feature_path, get_sanitized_path, get_coded_order_path
from ..utils import how2qa, msrvtt
from ..utils.dataset import PREDEFINED_PATHS
from ..utils.dataset import ray_get_with_tqdm, save_embedding

import numpy as np
import torch
import torch.nn as nn

class AvgPoolConv(nn.Module):
    def __init__(self):
        super(AvgPoolConv, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)

        # Initialize weights to 1/256 for averaging
        nn.init.constant_(self.conv.weight, 1/256)

    def forward(self, x):
        return self.conv(x)

def get_blocked_masks(frames, history=3, var_threshold=16, block_threshold=0.5):
    backsub = backSub = cv2.createBackgroundSubtractorMOG2(
            history=history, # default 500
            varThreshold=var_threshold, # default 16
            detectShadows=False
    )
    avg_pool = AvgPoolConv()

    fg_masks = []
    for frame in frames:
        fg = backsub.apply(frame)
        fg_mask = (fg > 0).astype(np.uint8)
        fg_masks.append(fg_mask)

    fg_masks = np.stack(fg_masks)

    fg_masks = torch.tensor(fg_masks, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        blocked_masks = avg_pool(fg_masks)
    blocked_masks = blocked_masks > block_threshold
    blocked_masks = blocked_masks.detach().numpy()
    return blocked_masks

def any_in(mb_type, symbol_list):
    return any(symbol in mb_type for symbol in symbol_list)

def convert_mb_cova(mb_type):
    '''Based on the following repos:
    - https://github.com/FFmpeg/FFmpeg/blob/918de766f545008cb1ab3d62febe71fa064f8ca7/libavcodec/mpegutils.c#L196
    '''
    if any_in(mb_type, ['I', 'i', 'A']):  # IS_INTRA
        return 6
    elif any_in(mb_type, ['d', 'D', 'g', 'S']): # IS_SKIP or IS_DIRECT
        return 1
    # elif any_in(mb_type, ['']) # IS_SUB_4X4
    #     return 5
    elif any_in(mb_type, ['|', '-']): # IS_16X8 or IS_8X16
        return 3
    elif '+' in mb_type: # IS_8X8
        return 4
    else:# IS_16X16
        return 2
    # else:
    #     return 0


def convert_mb_types(mb_types, width, height):
    converted_mb_types = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            converted_mb_types[i][j] = convert_mb_cova(mb_types[i][j])
    return converted_mb_types

def get_mb_types(video_path, width, height, verbose=False):
    CMD = f"ffmpeg -threads 1 -debug 'mb_type' -i {video_path} -f null -"

    if verbose:
        print(CMD)

    handle = subprocess.Popen(
        shlex.split(CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    com = handle.communicate()

    output = com[1].decode()

    frames = []

    idx = 0
    lines = output.splitlines()
    lines = list(filter(lambda line: line.startswith('[h264'), lines))
    lines = list(filter(lambda line: 'nal_unit_type' not in line, lines))
    debug_lines = '\n'.join(lines)
    while True:
        if idx >= len(lines):
            break
        line = lines[idx]
        if 'New frame' in line:
            frame = []
            idx += 1
            # if len(lines[idx].split()) == 7:
            #     print(lines[idx])
            #     # Newer version of ffmpeg added the following format
            #     # [h264 @ 0x556b1a63f5c0]     0           64          128         192
            #     # [h264 @ 0x556b1a63f5c0]   0 >  I  i  I  I  I  I  >  >  I  S  I  I  I  S  i
            #     expected_row_len = width + 4
            #     idx += 1
            # else:

            expected_row_len = width + 3

            for i in range(height):
                line = lines[idx]
                splitted = line.strip().split()
                assert len(splitted) == expected_row_len, f'{video_path} line {idx} has {len(splitted)} elements instead of {expected_row_len}: {line}\n{debug_lines}'
                frame.append(splitted[-width:])
                idx += 1

            frames.append(convert_mb_types(frame, width, height))
        else:
            idx += 1

    return np.array(frames)

def get_qps(video_path, width, height, verbose=False):
    CMD = f"ffmpeg -threads 1 -debug 'qp' -i {video_path} -f null -"

    if verbose:
        print(CMD)

    handle = subprocess.Popen(
        shlex.split(CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    com = handle.communicate()

    output = com[1].decode()

    frames = []

    idx = 0
    lines = output.splitlines()
    lines = list(filter(lambda line: line.startswith('[h264'), lines))
    lines = list(filter(lambda line: 'nal_unit_type' not in line, lines))
    debug_lines = '\n'.join(lines)
    while True:
        if idx >= len(lines):
            break

        line = lines[idx]
        if 'New frame' in line:
            idx += 1
            frame = []
            # Due to case like the below, counting words doesn't work. Just use ffmpeg 4.3
            # [h264 @ 0x55fae821fd00] 121213132222 9 9 711172019112114
            # if len(lines[idx].split()) == 7:
            #     print(lines[idx])
            #     raise NotImplementedError('For some reason, the code below doesn\'t work. Use ffmpeg version 4.3')
            #     # Newer version of ffmpeg added the following format
            #     # [h264 @ 0x556b1a63f5c0]     0           64          128         192
            #     # [h264 @ 0x556b1a63f5c0]   0 >  I  i  I  I  I  I  >  >  I  S  I  I  I  S  i
            #     expected_header_len = 4
            #     idx += 1
            # else:
            expected_header_len = 3
            for i in range(height):
                line = lines[idx]
                qp_str = ' '.join(line.split()[expected_header_len:])
                qps = []

                try:
                    for j in range(width):
                        qps.append(int(qp_str[2*j: 2*j+2]))
                except Exception as e:
                    print(f'{e} in {video_path} line {idx}: {qp_str}\n{debug_lines}')
                    exit()

                assert len(qps) == width, f'{video_path} line {idx} has {len(qps)} elements instead of {width}: {qp_str}\n{debug_lines}'
                frame.append(qps)
                idx += 1
            frames.append(frame)
        else:
            idx += 1

    return np.array(frames)

def get_motion_vectors(video_path, width, height):
    from mvextractor.videocap import VideoCap
    cap = VideoCap()
    ret = cap.open(str(video_path))
    if not ret:
        print('Failed to open video')
        print(video_path)

    motion_vectors = []
    frames = []
    while True:
        ret, frame, mv, frame_type, timestamp = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        converted_motion_vectors = np.zeros((height, width, 2), dtype=np.float32)
        vector_cnts = np.zeros((height, width), dtype=np.int32)

        start_point = mv[:, 3:5]
        end_point = mv[:, 5:7]

        mv = (end_point - start_point) / 16
        start_coord = (start_point - 8) // 16

        for i in range(len(mv)):
            x, y = start_coord[i]
            if x >= width or y >= height or x < 0 or y < 0:
                continue
            converted_motion_vectors[y, x] += mv[i]
            vector_cnts[y, x] += 1

        vector_cnts[vector_cnts == 0] = 1
        motion_vectors.append(converted_motion_vectors / vector_cnts[:, :, None])

    cap.release()

    motion_vectors = np.stack(motion_vectors)
    return motion_vectors, frames

def get_coded_order(video_path):
    CMD = f'ffprobe -show_frames -select_streams v:0 {video_path}'

    with subprocess.Popen(
        shlex.split(CMD),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as handle:
        com = handle.communicate()
        output = com[0].decode()

    coded_order = []
    for line in output.splitlines():
        if 'coded_picture_number' in line:
            splitted = line.split('=')
            coded_order.append(int(splitted[1]))

    coded_order = np.argsort(coded_order)

    return coded_order

@ray.remote(num_cpus=1)
def run_extract_compressed(
    video_id,
    video_path,
    feature_dir,
    width,
    height,
    dry_run,
    use_feature_v2=False,
    extract_coded_order_only=False,
    skip_pixel_check=False,
    start=None
):
    if not skip_pixel_check:
        first_pixel_path = get_feature_path(feature_dir, video_id, 'i', 0, use_v2=use_feature_v2)
        if not Path(first_pixel_path).exists():
            print(f'Skipping {video_id} because {first_pixel_path} does not exist')
            return

    cap = cv2.VideoCapture(str(video_path))
    is_opened = cap.isOpened()
    cap.release()
    if not is_opened:
        print(f'Failed to open {video_path}')
        return

    mb_types = get_mb_types(video_path, width, height, verbose=False)
    mvs, frames = get_motion_vectors(video_path, width, height)
    qps = get_qps(video_path, width, height)

    compressed = np.concatenate(
        (
            np.expand_dims(mb_types, -1),
            mvs,
            np.expand_dims(qps, -1)
        ),
        axis=-1,
        dtype=np.float32
    )
    compressed = compressed.transpose(0, 3, 1, 2)

    if extract_coded_order_only:
        coded_order = get_coded_order(video_path)
        output_path = get_coded_order_path(feature_dir, video_id)
        if dry_run:
            print(f'Would save to {output_path}')
        else:
            save_embedding(coded_order, output_path)
    # blocked_masks = get_blocked_masks(frames)
    else:
        if start == None:
            start = 0
        for idx, c in enumerate(compressed):
            per_idx_path = get_feature_path(feature_dir, video_id, 'c', start + idx, use_v2=use_feature_v2)
            if dry_run:
                print(f'Would save to {per_idx_path}')
            else:
                save_embedding(c, per_idx_path)

def extract_compressed_features(
        video_ids,
        video_paths,
        feature_dir,
        width,
        height,
        dry_run=False,
        use_feature_v2=False,
        extract_coded_order_only=False,
        skip_pixel_check=False,
        starts=None,
    ):
    # Create output directory
    # if not dry_run:
    #     Path(feature_dir).mkdir(parents=True, exist_ok=True)

    ret = []

    for idx, (video_id, video_path) in enumerate(zip(video_ids, video_paths)):
        r = run_extract_compressed.remote(
            video_id,
            video_path,
            feature_dir,
            width,
            height,
            dry_run,
            use_feature_v2=use_feature_v2,
            extract_coded_order_only=extract_coded_order_only,
            skip_pixel_check=skip_pixel_check,
            start=None if starts is None else starts[idx]
        )
        ret.append(r)

    ray_get_with_tqdm(ret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("DATASET", choices=get_available_datasets())
    parser.add_argument("BASE_MODEL_NAME")
    parser.add_argument("FPS", type=int)
    parser.add_argument("SPLIT", choices=get_available_splits())
    parser.add_argument("--dry-run", action='store_true')
    parser.add_argument("--extract-coded-order-only", action='store_true')
    parser.add_argument("--skip-pixel-check", action='store_true')
    parser.add_argument("--use-start-end", action='store_true')
    parser.add_argument("--dir-key", type=str, default='feature_dir')
    args = parser.parse_args()

    starts = None
    use_feature_v2 = False

    if args.DATASET == 'how2qa':
        assert args.use_start_end, 'how2qa requires --use-start-end'
        video_ids, video_paths, start_ends = how2qa.get_youtube_ids_and_paths(
            args.SPLIT,
            fps=args.FPS,
            return_time=True
        )
        starts = list(int(s_e[0] * args.FPS) for s_e in start_ends)
        mb_width = 16
        mb_height = 16
    elif args.DATASET == 'msrvtt':
        video_ids, video_paths, start_ends = msrvtt.get_video_ids_and_paths(args.SPLIT, fps=args.FPS)
        mb_width = 14
        mb_height = 14
    else:
        raise NotImplementedError

    feature_dir = get_feature_dir(args.DATASET, args.BASE_MODEL_NAME, args.FPS, args.SPLIT, dir_key=args.dir_key)

    if not args.dry_run:
        Path(feature_dir).mkdir(parents=True, exist_ok=True)

    if len(video_paths) > 0:
        extract_compressed_features(
            video_ids,
            video_paths,
            feature_dir,
            mb_width,
            mb_height,
            dry_run=args.dry_run,
            use_feature_v2=use_feature_v2,
            extract_coded_order_only=args.extract_coded_order_only,
            skip_pixel_check=args.skip_pixel_check,
            starts=starts
        )
