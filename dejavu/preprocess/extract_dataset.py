import argparse
import torch

from dejavu.utils.scripts import run_batch_for_train_model


from ..model.diffrate.diffrate import create_diffrate_model
from ..model.reuse.opt_attention import create_opt_attn_model
from ..train.reuse.model import ReuseModel
from ..utils.visualize import visualize_reuse_maps_from_inference
from ..utils import (
    PREDEFINED_PATHS,
    get_feature_path,
    save_embedding,
    get_diffrate_prune_merge,
    get_feature_dir_diffrate,
    get_feature_dir_reuse,
    get_feature_dir_cmc,
    get_available_datasets
)
import numpy as np
from ..dataset import (
    How2qaExtractDataset, MsrvttExtractDataset,
    How2qaReuseExtractDataset, MsrvttReuseExtractDataset,
)
from tqdm import tqdm 
from pathlib import Path
import torch.cuda.nvtx as nvtx

CACHE_DIR = PREDEFINED_PATHS['root']['cache']

def parse_batch(batch, args):
    video_id = is_new_video = frame_idxs = original_pixels = pixel_values = original_outputs = compressed = None
    if args.mode == 'reuse':
        if args.visualize_dir is not None:
            video_id, is_new_video, frame_idxs, original_pixels, pixel_values, original_outputs, compressed = batch
            original_pixels = original_pixels.squeeze(0)
            original_outputs = original_outputs.squeeze(0)
        else:
            video_id, is_new_video, frame_idxs, pixel_values, original_outputs, compressed = batch
            original_outputs = original_outputs.squeeze(0)

        pixel_values = pixel_values.squeeze(0)
        compressed = compressed.squeeze(0)
    else:
        video_id, frame_idxs, pixel_values, original_outputs = items

    ret = [video_id, is_new_video, frame_idxs, original_pixels, pixel_values, original_outputs, compressed]
    for idx, item in enumerate(ret):
        if item is not None and hasattr(item, 'cuda'):
            ret[idx] = item.cuda()

    return ret


def restore_hh(hidden_states, maps):
    assert len(hidden_states) == len(maps) + 1

    dim = hidden_states[-1].shape[-1]

    final_map = maps[0].clone()
    final_prune_mask = maps[0] == 512

    # First, unravel the final map
    for m in maps[1:]:
        final_map = torch.gather(m, 1, final_map)
        prune_mask = final_map == 512
        final_map[prune_mask] = 0 # Prevent pruned states from Out of order
        final_prune_mask = final_prune_mask | prune_mask

    hh = torch.gather(
        hidden_states[-1],
        axis=1,
        index=final_map.unsqueeze(-1).expand(-1, -1, dim)
    )

    hh[prune_mask] = 0

    return hh


def update_pbar(pbar, sim, reuse_rates):
    if sim is None:
        return
    sim = sim.cpu().tolist()
    reuse_rates = reuse_rates.cpu().tolist()

    # Format each element in sim and reuse_rates to three decimal places
    formatted_sim = [f"{x:.3f}" for x in sim]
    formatted_reuse_rates = [f"{x:.3f}" for x in reuse_rates]

    # Create the description string with the formatted lists
    desc = f'sim: {formatted_sim}, reuse: {formatted_reuse_rates}'
    pbar.set_description(desc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original', 'reuse', 'diffrate', 'cmc'], required=True)
    parser.add_argument('--dataset', choices=get_available_datasets(), help='Dataset to extract from', required=True)
    parser.add_argument('--fps', type=int, default=2, help='Override FPS instead of default')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dry-run', action='store_true', help='Do not save features')
    parser.add_argument('--debug-no-reuse', action='store_true', help='Disable reuse for debugging')
    parser.add_argument('--use-start-end', action='store_true', help='should be enabled when extract how2qa diffrate')
    parser.add_argument("--target-features", type=str, default='o,hh', help='Target features to extract (comma separated)')
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity level. Use -v for basic verbosity and -vv for more detailed."
    )
    parser.add_argument('--visualize-dir', type=str, default=None)

    # DiffRate related
    parser.add_argument('--diffrate-target-flops', type=str, default=None)
    parser.add_argument('--diffrate-finetuned', type=str, default=None)
    parser.add_argument('--diffrate-restore', action='store_true')
    # Reuse related
    parser.add_argument('--reuse-model-name', type=str, default=None)
    parser.add_argument('--reuse-refresh-interval', type=int, default=0)
    # CMC related
    parser.add_argument(
        '--cmc-threshold', type=float, default=None,
        help='Threshold for CMC model, note that threshold will be negated for implementation'
    )
    parser.add_argument('--measure-throughput', action='store_true')

    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world-size', type=int, default=None)

    args = parser.parse_args()

    if args.mode == 'reuse' or args.mode == 'cmc' or args.mode == 'eventful':
        assert args.reuse_model_name is not None, "Must specify reuse model name for Reuse mode."
        if not args.measure_throughput:
            print("Setting batch size to 1 for Reuse mode")
            args.batch_size = 1
    elif args.mode == 'diffrate':
        assert args.diffrate_target_flops is not None, "Must specify target flops for Diffrate mode."
    else:
        assert args.mode == 'original', "Unknown mode."

    args.target_features = args.target_features.split(',')

    return args


def initialize_dataset(args, base_model_name, ds_kwargs={}):
    dataset_classes = {
        'how2qa': (How2qaReuseExtractDataset, How2qaExtractDataset),
        'msrvtt': (MsrvttReuseExtractDataset, MsrvttExtractDataset),
    }

    if args.dataset not in dataset_classes:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    ReuseDataset, OriginalDataset = dataset_classes[args.dataset]

    if args.mode == 'reuse':
        DatasetClass = ReuseDataset
        print('Setting return_compress to True for Reuse mode')
        ds_kwargs['return_compressed'] = True
    elif args.mode == 'diffrate':
        DatasetClass = OriginalDataset
    elif args.mode == 'cmc' or args.mode == 'eventful':
        DatasetClass = ReuseDataset
        print('Setting is_sequential to True for Reuse mode')
        ds_kwargs['is_sequential'] = True
    elif args.mode == 'original':
        DatasetClass = OriginalDataset
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    ds = DatasetClass(base_model_name, args.fps, **ds_kwargs)

    if args.rank is not None and args.world_size is not None:
        if DatasetClass == ReuseDataset:
            start_idx, end_idx = ds.return_range_per_rank(args.rank, args.world_size)
            idxs = list(range(start_idx, end_idx))
        else:
            idxs = list(range(args.rank, len(ds), args.world_size))

        ds = torch.utils.data.Subset(
            ds,
            idxs
        )

    return ds

def get_base_model_and_split(args):
    dataset_base_info = {
        'how2qa': ('openai/clip-vit-large-patch14', 'frozenbilm'),
        'msrvtt': ('openai/clip-vit-base-patch16', 'test'),
    }

    if args.dataset not in dataset_base_info:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    base_model_name, split = dataset_base_info[args.dataset]
    return base_model_name, split



def create_model(args, base_model_name, split):
    if args.mode == 'original':
        raise NotImplementedError("Original mode is not implemented yet.")

    elif args.mode == 'diffrate':
        diffrate_model_name = args.diffrate_finetuned if args.diffrate_finetuned else f'original-{args.diffrate_target_flops}'
        print(f"Using Diffrate model: {diffrate_model_name}")

        prune, merge = get_diffrate_prune_merge(
            args.dataset, f"diffrate/{args.dataset}/{args.diffrate_target_flops}"
        )
        if args.diffrate_restore:
            print("Pruning is disabled for restore mode")
            prune = None
            merge = list(max(p, m) for p, m in zip(prune, merge)) if prune else None

        model = create_diffrate_model(
            base_model_name,
            prune,
            merge,
            finetuned_model=args.diffrate_finetuned,
        )
        feature_dir = get_feature_dir_diffrate(
            args.dataset, base_model_name, args.fps, split, diffrate_model_name
        )
        kwargs = {}
    
    elif args.mode == 'reuse':

        model = create_opt_attn_model(
            args.reuse_model_name,
            cache_dir=CACHE_DIR,
        )
        feature_dir = get_feature_dir_reuse(
            args.dataset,
            base_model_name,
            args.fps,
            split,
            args.reuse_model_name,
            refresh_interval=args.reuse_refresh_interval
        )
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_patch_one_side = 224 // model.config.patch_size
        N = num_patch_one_side ** 2 + 1
        kwargs = {
            'reference_caches': torch.zeros(
                (num_hidden_layers, args.batch_size, N, hidden_size),
                device='cuda',
                dtype=torch.float32
            ),
            'hqkv_caches': torch.zeros(
                (num_hidden_layers, 4, args.batch_size, N, hidden_size),
                device='cuda',
                dtype=torch.float32
            ),
            'reference_type': torch.nn.functional.one_hot(
                torch.tensor([0, 1, 2, 2], device='cuda').unsqueeze(1).expand(-1, args.batch_size),
            )
        }

    elif args.mode == 'cmc':
        assert args.cmc_threshold is not None, "Must specify CMC threshold."
        feature_dir = get_feature_dir_cmc(
            args.dataset, base_model_name, args.fps, split, threshold=args.cmc_threshold
        )

        config = PREDEFINED_PATHS['train'][f'cmc-{args.dataset}']
        config['decision_hyperparam'] = -args.cmc_threshold

        model = ReuseModel(**config)
        kwargs = {}

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    model = model.eval().cuda()

    if args.compile:
        model = torch.compile(model)
    
    return model, feature_dir, kwargs


def cosine_similarity(x, y):
    if x is None or y is None:
        return None
    x = x.squeeze(1)
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)

def extract_features(
        model,
        pixel_values,
        compressed,
        is_new_video,
        args,
        model_kwargs,
        is_training_model=False,
        return_timers=False
    ):
    if is_new_video is not None:
        is_new_video = all(is_new_video)

    if return_timers:
        starter = torch.cuda.Event(enable_timing=True)
        starter.record()

    if not is_training_model:
        inference_outputs = model(
            pixel_values=pixel_values,
            compressed=compressed,
            disable_reuse=args.debug_no_reuse or is_new_video,
            output_hidden_states=True,
            output_maps=True,
            **model_kwargs
        )

        if args.mode == 'diffrate':
            hh = restore_hh(
                inference_outputs.hidden_states[:-1],
                inference_outputs.maps[:-1]
            )
        else:
            # Note that -1 contains the second from last hidden state
            hh = inference_outputs.hidden_states[-1].view(4, -1, model.config.hidden_size)
        outputs = inference_outputs.image_embeds

    else:
        if is_new_video:
            cached_states_for_next_batch_0123 = None
        else:
            cached_states_for_next_batch_0123 = model_kwargs['cached_states_for_next_batch_0123']

        output, maps, cached_states_for_next_batch = run_batch_for_train_model(
            model,
            pixel_values=pixel_values,
            cached_states_from_prev_batch=cached_states_for_next_batch_0123,
            **model_kwargs
        )

        cached_states_for_next_batch_0123 = []
        for cached_states in cached_states_for_next_batch:
                if cached_states is not None:
                    cached_states = [c[:, -197:] for c in cached_states]
                cached_states_for_next_batch_0123.append(cached_states)
        model_kwargs['cached_states_for_next_batch_0123'] = cached_states_for_next_batch_0123

        hh = None

    if return_timers:
        ender = torch.cuda.Event(enable_timing=True)
        ender.record()
        return outputs, hh, inference_outputs, starter, ender

    return outputs, hh, inference_outputs
    


def return_features(model, dataset, query_id, args, kwargs):
    start_idx, end_idx = dataset.get_video_range(query_id)

    outputs = {}
    hhs = {}

    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            items = dataset[idx]
            video_id, is_new_video, frame_idxs, original_pixels, pixel_values, original_outputs, compressed = parse_batch(items, args)

            assert video_id == query_id

            o, hh, inference_outputs = extract_features(model, pixel_values, compressed, is_new_video, args, kwargs)

            for stack_idx, frame_idx in enumerate(frame_idxs):
                frame_idx = frame_idx.item()
                if frame_idx not in outputs:
                    outputs[frame_idx] = o[stack_idx]
                    hhs[frame_idx] = hh[stack_idx]

    # Sort by keys and return as list
    outputs = [outputs[i] for i in sorted(outputs.keys())]
    hhs = [hhs[i] for i in sorted(hhs.keys())]

    return outputs, hhs


if __name__ == '__main__':
    import termplotlib as tpl
    args = parse_args()

    base_model_name, split = get_base_model_and_split(args)

    ds_kwargs = {
        'return_pixel_values': args.visualize_dir is not None,
        'return_input_values': True,
        'return_hidden_states': False,
        'return_output_states': True,
        'return_compressed': False,
        'use_start_end': args.use_start_end,
    }

    use_v2 = False

    if args.mode == 'reuse':
        ds_kwargs['refresh_interval'] = args.reuse_refresh_interval

    ds = initialize_dataset(args, base_model_name, ds_kwargs)

    model, feature_dir, model_kwargs = create_model(args, base_model_name, split)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    if args.visualize_dir is not None:
        visualize_dir = Path(args.visualize_dir)

    sim_list = []
    reuse_rate_list = []
    frame_idx_list = []
    is_first_video = True
    batch_idx = 0

    if args.measure_throughput:
        starters = []
        enders = []

    def save_embedding_local(embedding, path):
        if not args.dry_run:
            path.parent.mkdir(exist_ok=True, parents=True)
            save_embedding(embedding, path)
        elif args.verbose >= 2:
            print(f'DRYRUN: Would save to {path}')


    with tqdm(dataloader) as pbar:
        for batch_idx, items in enumerate(pbar):
            video_ids, is_new_video, frame_idxs, original_pixels, pixel_values, original_outputs, compressed = parse_batch(items, args)

            with torch.no_grad():
                if args.measure_throughput and batch_idx == len(dataloader) - 1:
                    # Skipping the last batch for throughput measurement
                    break

                items = extract_features(
                    model,
                    pixel_values,
                    compressed,
                    is_new_video,
                    args,
                    model_kwargs,
                    return_timers=args.measure_throughput
                )
                if args.measure_throughput:
                    outputs, hh, inference_outputs, starter, ender = items
                    starters.append(starter)
                    enders.append(ender)
                    continue

                outputs, hh, inference_outputs = items

                sim = cosine_similarity(outputs, original_outputs)
                reuse_rates = None

                if args.mode == 'reuse':
                    reuse_rates = [
                        m.float().mean(dim=(1, 2)) for m in inference_outputs.maps if m is not None
                    ]
                    reuse_rates = torch.stack(reuse_rates, dim=-1).mean(dim=-1)

            outputs = outputs.cpu()

            # Log metrics
            if args.mode == 'reuse' and is_new_video:
                batch_idx = 0
                if is_first_video:
                    is_first_video = False
                    first_batch_sim = sim
                    first_batch_reuse_rate = reuse_rates
                
                elif args.verbose:
                    print("=" * 80)
                    print(f"Video ID: {video_id}") # Is video_id of previous batch yet
                    print(f"First batch cosine similarity: {first_batch_sim}")
                    print(f"First batch reuse rate: {first_batch_reuse_rate}")
                    print(f"Rest avg. cosine similarity: {np.array(sim_list[1:]).mean()}")
                    print(f"Rest avg. reuse rate: {np.array(reuse_rate_list[1:]).mean()}")

                    # Sort y by x
                    frame_idx_list, sim_list, reuse_rate_list = zip(*sorted(zip(frame_idx_list, sim_list, reuse_rate_list)))

                    fig = tpl.figure()
                    fig.plot(
                        x=np.array(frame_idx_list),
                        y=np.array(sim_list),
                        label='Cosine similarity',
                        width=140, height=10
                    )
                    fig.plot(
                        x=np.array(frame_idx_list),
                        y=np.array(reuse_rate_list),
                        label='Reuse rate',
                        width=140, height=10
                    )
                    fig.show()

                    # Reset the lists
                    sim_list = []
                    reuse_rate_list = []
                    frame_idx_list = []


            if args.mode == 'reuse':
                update_pbar(pbar, sim, reuse_rates)

            for i in range(len(outputs)):
                if args.mode == 'reuse':
                    video_id = video_ids[0]
                    frame_idx = frame_idxs[0, i]
                else:
                    video_id = video_ids[i]
                    frame_idx = frame_idxs[i]

                if int(frame_idx) < 0:
                    continue

                if isinstance(frame_idx, torch.Tensor):
                    frame_idx = frame_idx.item()
                
                o_path = get_feature_path(feature_dir, video_id, 'o', frame_idx, use_v2=use_v2)
                hh_path = get_feature_path(feature_dir, video_id, 'hh', frame_idx, use_v2=use_v2)

                if 'o' in args.target_features:
                    save_embedding_local(outputs[i], o_path)
                if 'hh' in args.target_features:
                    save_embedding_local(hh[i], hh_path)

                if args.mode == 'reuse' and args.verbose:
                    if frame_idx not in frame_idx_list:
                        sim_list.append(sim[i].cpu())
                        reuse_rate_list.append(reuse_rates[i].cpu())
                        frame_idx_list.append(frame_idx)

            if args.visualize_dir is not None:
                save_path = visualize_dir / f'{video_id}/{batch_idx}.jpeg'
                visualize_reuse_maps_from_inference(
                    original_pixel_values=original_pixels,
                    frame_idxs=frame_idxs.squeeze(0),
                    output=inference_outputs,
                    save_path=save_path
                )

            batch_idx += 1

    if args.measure_throughput:
        torch.cuda.synchronize()

        total_time = 0
        for starter, ender in zip(starters, enders):
            total_time += starter.elapsed_time(ender) / 1000

        data_size = len(ds) * 4
        print(f"Total time elapsed: {total_time:.2f} s")
        print(f"Data size: {data_size}")
        print(f"Throughput: {data_size / total_time:.2f} samples/s")
