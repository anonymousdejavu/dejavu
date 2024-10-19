import ray
import argparse
import numpy as np
import subprocess
import shutil


def extract_dataset(dataset, fps, threshold, output_path):
    CMD = f'python -m dejavu.model.cmc.extract --dataset {dataset} --fps {fps} --threshold {threshold} --is-sweep --dry-run'
    with open(output_path, 'w') as f:
        com = subprocess.Popen(CMD, shell=True, stdout=f, stderr=subprocess.STDOUT)
        com.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['how2qa', 'msrvtt'], required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--base-model-name', choices=['openai/clip-vit-large-patch14', 'openai/clip-vit-base-patch16'], default=None)
    parser.add_argument('--fps', type=int, default=None, help='Override FPS instead of default')
    parser.add_argument('--threhsold-min', type=float, default=0)
    parser.add_argument('--threhsold-max', type=float, default=160)

    args = parser.parse_args()

    if args.base_model_name is None:
        if args.dataset == 'how2qa':
            args.base_model_name = 'openai/clip-vit-large-patch14'
        else:
            args.base_model_name = 'openai/clip-vit-base-patch16'

    # Base model:  ??? MiB / 24576MiB
    # Large model: 2455MiB / 24576MiB
    if args.base_model_name == 'openai/clip-vit-large-patch14':
        num_gpus_per_task = 24576 // 2455
        num_layers = 24
    elif args.base_model_name == 'openai/clip-vit-base-patch16':
        raise NotImplementedError("Not implemented yet")
        num_layers = 12

    extract_dataset_ray = ray.remote(num_gpus=num_gpus_per_task)(extract_dataset)

    args.is_sweep = True
    args.dry_run = True

    # Perform grid search
    

