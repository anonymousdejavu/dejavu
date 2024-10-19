import sys
import argparse
import torch

from ...train.reuse.model import ReuseModel
from ...utils import (
    PREDEFINED_PATHS,
    get_feature_path,
    save_embedding,
    get_diffrate_prune_merge,
    get_feature_dir_diffrate,
    get_feature_dir_reuse,
    get_feature_dir_cmc
)
from ...utils.scripts import run_batch_for_train_model

from ...dataset import How2qaDataset, MsrvttDataset
from tqdm import tqdm 

def main(args):
    if args.dataset == 'how2qa':
        BASE_MODEL_NAME='openai/clip-vit-large-patch14'
        SPLIT='frozenbilm'
        if args.fps is None:
            args.fps = 1
        Dataset = How2qaDataset
    elif args.dataset == 'msrvtt':
        BASE_MODEL_NAME = 'openai/clip-vit-base-patch16'
        SPLIT='test'
        if args.fps is None:
            args.fps = 1
        Dataset = MsrvttDataset
    else:
        raise NotImplementedError

    if args.dataset == 'how2qa':
        ds = Dataset(
            split=SPLIT,
            base_model_name=BASE_MODEL_NAME,
            fps=args.fps,
            return_output_states=False,
            use_start_end=not args.disable_start_end
        )
    else:
        ds = Dataset(
            split=SPLIT,
            base_model_name=BASE_MODEL_NAME,
            return_output_states=False,
            fps=args.fps
        )

    if args.is_sweep:
        # Sweep over 10% of the dataset
        assert args.rank is not None and args.world_size is not None, "Rank and world size must be specified for sweep"
        print(f"Sweeping with 5% of the dataset")
        ds = ds[:int(len(ds)*0.05)]

    if args.rank is not None and args.world_size is not None:
        ds = torch.utils.data.Subset(
            ds,
            list(range(args.RANK, len(ds), args.WORLD_SIZE))
        )

    feature_dir = get_feature_dir_cmc(
        args.dataset,
        BASE_MODEL_NAME,
        args.fps,
        SPLIT,
        threshold=args.cmc_threshold,
        reuse_start_before_mlp=args.reuse_start_before_mlp
    )

    config = PREDEFINED_PATHS['train'][f'cmc/{args.dataset}']
    if args.dataset == 'how2qa':
        config['decision_hyperparam'] = [-args.cmc_threshold] * 24
    else:
        config['decision_hyperparam'] = [-args.cmc_threshold] * 12

    if args.reuse_start_before_mlp:
        config['reuse_start'] = 'before_mlp'

    model = ReuseModel(**config)
    model = model.eval()
    model = model.to(args.device)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    total_sim = 0
    total_reuse_rate = 0
    total_layerwise_reuse_rates = None
    cnt = 0

    desc='Extracting features'
    pbar = tqdm(total=len(dataloader), desc=desc, disable=args.is_sweep)

    prev_frame_idx = -sys.maxsize - 1
    prev_video_id = -sys.maxsize - 1

    for batch_idx, (frame_idx, pixel_values) in enumerate(dataloader):
        pixel_values = pixel_values.to(args.device).unsqueeze(1)
        # original_outputs = original_outputs.to(args.device).unsqueeze(1)

        video_id = ds.id_idx[batch_idx][0]

        is_new_video = False

        if prev_frame_idx + 1 != frame_idx and prev_video_id != video_id:
            is_new_video = True

        if is_new_video:
            cached_states_for_next_batch_0123 = None

        with torch.no_grad():
            outputs, maps, cached_states_for_next_batch = run_batch_for_train_model(
                model, 
                pixel_values=pixel_values,
                cached_states_from_prev_batch=cached_states_for_next_batch_0123,
            )
            cached_states_for_next_batch_0123 = []
            for cached_states in cached_states_for_next_batch:
                if cached_states is not None:
                    cached_states = [c[:, -197:] for c in cached_states]
                cached_states_for_next_batch_0123.append(cached_states)

            # skip the first idx.
            if not is_new_video:
                # Shape: B, F
                # sim = cosine_sim(outputs, original_outputs)
                # Shape: B, F
                # maps: 1, 1, 11, 197
                # mean_reuse_rates: 1, 1
                mean_reuse_rates = maps.mean(dim=(2, 3))
                layerwise_reuse_rates = maps.mean(dim=(3))

                # total_sim += sim.sum().item()
                total_reuse_rate += mean_reuse_rates.sum().item()
                if total_layerwise_reuse_rates is None:
                    total_layerwise_reuse_rates  = layerwise_reuse_rates.sum(dim=(0,1))
                else:
                    total_layerwise_reuse_rates += layerwise_reuse_rates.sum(dim=(0,1))
                # cnt += sim.numel()
                cnt += mean_reuse_rates.numel()

            o_path = get_feature_path(feature_dir, video_id, 'o', frame_idx.item())
            o_embedding = outputs[0, 0]
            if not args.dry_run:
                o_path.parent.mkdir(exist_ok=True, parents=True)
                save_embedding(o_embedding, o_path)
            else:
                print(f'DRYRUN: Would save tensor of size {o_embedding.shape} to {o_path}')
        if cnt > 0:
            # desc = f'Avg. sim: {total_sim / cnt:.2f}, Avg. reuse rate: {total_reuse_rate / cnt:2.2%}'
            desc = f'Avg. reuse rate: {total_reuse_rate / cnt:2.2%}'
            pbar.set_description(desc)
        pbar.update()

        prev_frame_idx = frame_idx
        prev_video_id = video_id

    print(desc)

    total_layerwise_reuse_rates /= cnt    
    total_layerwise_reuse_rates = total_layerwise_reuse_rates.tolist()
    num_layer = len(total_layerwise_reuse_rates)
    print(f"total layerwise reuse rate: ", end="[")
    for i in range(num_layer):
        value = total_layerwise_reuse_rates[i]

        if i < num_layer-1:
            print(value, end=",")
        else:
            print(value, end="]\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['how2qa', 'msrvtt'], required=True)
    parser.add_argument('--fps', type=int, default=None, help='Override FPS instead of default')
    parser.add_argument('--dry-run', action='store_true', help='Do not save features')
    parser.add_argument('--debug-no-reuse', action='store_true', help='Disable reuse for debugging')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--is-sweep', action='store_true')
    parser.add_argument('--disable-start-end', action='store_true')
    parser.add_argument('--reuse-start-before-mlp', action='store_true')
    parser.add_argument(
        '--cmc-threshold', type=float, default=140,
        help='Threshold for CMC model, note that threshold will be negated for implementation'
    )

    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)

    args = parser.parse_args()

    main(args)