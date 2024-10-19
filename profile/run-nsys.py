#!/usr/bin/env python
import torch
import json

import sys
if '/workspace' not in sys.path:
    sys.path.append('/workspace')

import torch.cuda.nvtx as nvtx

from dejavu.dataset import How2qaEvaluationDataset, MsrvttEvaluationDataset
from dejavu.utils import PREDEFINED_PATHS
from dejavu.model.clip import CLIPVisionModelWithProjection
from dejavu.utils.diffrate import get_diffrate_prune_merge
# from dejavu.model.reuse.optimized_sim import create_sim_model
from dejavu.model.reuse.opt_attention import create_opt_attn_model
from dejavu.model.diffrate.diffrate import create_diffrate_model
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
from itertools import cycle

# from kernl.model_optimization import optimize_model

BASE_MODEL_NAME='openai/clip-vit-base-patch16'
CACHE_DIR='/mnt/ssd1/cache'
checkpoint_path = PREDEFINED_PATHS['train']["checkpoint_dir"]

import torch
from torch import nn

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['original', 'opt_attn', 'diffrate'])
    parser.add_argument('--reuse-model-name', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--diffrate-target-flops', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-inference', type=int, default=100)
    parser.add_argument('--use-compile', action='store_true')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--use-bf16', action='store_true')
    args = parser.parse_args()

    print("-- experiment config")
    print(args)

    if args.dataset == "msrvtt":
        base_model_name = 'openai/clip-vit-base-patch16'
    elif args.dataset == "how2qa":
        base_model_name = 'openai/clip-vit-large-patch14'
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    nvtx.range_push("model_initialization")
    if args.mode == 'original':
        model = CLIPVisionModelWithProjection.from_pretrained(
            base_model_name,
            cache_dir=CACHE_DIR
        )
    elif args.mode == 'diffrate':
        prune, merge = get_diffrate_prune_merge(args.dataset, "diffrate/" + args.dataset + "/" + args.diffrate_target_flops)
        model = create_diffrate_model(
            base_model_name,
            prune,
            merge,
        )
    elif args.mode == 'opt_attn':
        model = create_opt_attn_model(
            args.reuse_model_name,
            cache_dir=CACHE_DIR,
        )
    else:
        raise ValueError(f'Unknown mode {args.mode}')

    patch_size = model.config.patch_size
    projection_dim = model.config.projection_dim
    hidden_size = model.config.hidden_size
    num_hidden_layers = model.config.num_hidden_layers
    num_patch_one_side = 224 // patch_size
    N = num_patch_one_side ** 2 + 1

    if args.mode == 'opt_attn':
        cache_kwargs = {
            'reference_caches': torch.zeros(
                (num_hidden_layers, args.batch_size, N, hidden_size),
                device=args.device,
                dtype=torch.float32 if not args.use_bf16 else torch.bfloat16
            ),
            'hqkv_caches': torch.zeros(
                (num_hidden_layers, 4, args.batch_size, N, hidden_size),
                device=args.device,
                dtype=torch.float32 if not args.use_bf16 else torch.bfloat16
            ),
            'reference_type': torch.nn.functional.one_hot(
                torch.tensor([0, 1, 2, 2], device='cuda').unsqueeze(1).expand(-1, args.batch_size),
            )
        }
    elif args.mode == 'original':
        use_compressed = False
    elif 'diffrate' in args.mode:
        use_compressed = False
    else:
        raise ValueError(f'Unknown mode {args.mode}')

    model = model.eval()

    if args.device == 'cuda':
        model = model.to(args.device)

    if args.use_compile:
        model = torch.compile(model)

    if args.use_bf16:
        model = model.bfloat16()
        for layer in model.vision_model.encoder.layers:
            if hasattr(layer, 'codecnet'):
                layer.codecnet = layer.codecnet.bfloat16()

    nvtx.range_pop() # model_initialization


    nvtx.range_push("data_transfer")
    if args.dataset == 'how2qa':
        DS = How2qaEvaluationDataset
        split = 'frozenbilm'
    elif args.dataset == 'msrvtt':
        DS = MsrvttEvaluationDataset
        split = 'test'
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    ds = DS(
        batch_size=args.batch_size,
        split=split,
        base_model_name=base_model_name,
        fps=args.fps,
    )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    nvtx.range_pop() # data_transfer

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    torch.cuda.cudart().cudaProfilerStart()
    iter_idx = 0
    warmup_iter = 2
    nvtx.range_push("model_inference")
    total_time = 0

    mean_reuse_rates = np.zeros((11,))
    cnt = 0

    with torch.no_grad():
        for batch in tqdm(cycle(dataloader), total=args.num_inference):
            if iter_idx == warmup_iter:
                starter.record()
            iter_idx += 1
            if iter_idx > args.num_inference:
                break

            frame_idxs, frames, _, compressed = batch
            frames = frames.view(-1, 3, 224, 224)
            compressed = compressed.view(-1, 4, 4, num_patch_one_side, num_patch_one_side)
            frames = frames.to(args.device)
            compressed = compressed.to(args.device)

            if args.use_bf16:
                frames = frames.bfloat16()
                compressed = compressed.bfloat16()
 
            if args.mode == 'opt_attn':
                output = model(
                    pixel_values=frames,
                    attention_mask=None,
                    causal_attention_mask=None,
                    compressed=compressed,
                    # output_maps=True,
                    **cache_kwargs
                )
                # for idx, m in enumerate(output.maps):
                #     if m is not None:
                #         mean_reuse_rate = m.float().mean()
                #         mean_reuse_rates[idx] += mean_reuse_rate
                # cnt += 1
                # print(mean_reuse_rates / cnt)
                # print(mean_reuse_rates.mean() / cnt)
            else:
                output = model(
                    pixel_values=frames,
                    attention_mask=None,
                    causal_attention_mask=None,
                )
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender) / 1000 # ms to s
        # total_time += time.time() - start
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    # TODO: write reuse rate?
    # data_size = (len(ds)-num_iter)*args.batch_size*4
    num_batch = args.num_inference - warmup_iter
    data_size = num_batch * 4 * args.batch_size

    result = {
            'mode': args.mode,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'time_elapsed': total_time,
            'data_size': data_size,
            'throughput': data_size/total_time,
        }

    from dejavu.utils.aux import rename_base_model
    if args.mode == 'opt_attn':
        fname = f"results/{args.mode}-{rename_base_model(args.reuse_model_name)}-{args.dataset}.json"
    elif args.mode == 'original':
        fname = f"results/{args.mode}-{args.dataset}.json"
    elif args.mode == 'diffrate':
        fname = f"results/{args.mode}-{args.dataset}-{args.diffrate_target_flops}.json"
    else:
        raise ValueError(f'Unknown mode {args.mode}')

    print(fname)

    json_file_path = Path(fname)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with json_file_path.open('w') as json_file:
        json.dump(result, json_file, indent=4)
    print(f"Time elapsed: {total_time}")
    print(f'Throughput: {data_size / total_time}')
