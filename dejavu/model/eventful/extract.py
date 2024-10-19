import sys
from .model import EventfulCLIP
from ...dataset import MsrvttDataset, How2qaDataset
from tqdm import tqdm
import argparse
import torch
from ...utils import PREDEFINED_PATHS, get_feature_path, save_embedding, get_feature_dir_eventful

def main(args):
    if args.dataset == 'how2qa':
        BASE_MODEL_NAME='openai/clip-vit-large-patch14'
        SPLIT='frozenbilm'
        if args.fps is None:
            args.fps = 2
        Dataset = How2qaDataset
    elif args.dataset == 'msrvtt':
        BASE_MODEL_NAME = 'openai/clip-vit-base-patch16'
        SPLIT='test'
        if args.fps is None:
            args.fps = 2
        Dataset = MsrvttDataset
    else:
        raise NotImplementedError

    if args.dataset == 'how2qa':
        ds = Dataset(
            split=SPLIT,
            base_model_name=BASE_MODEL_NAME,
            fps=args.fps,
            use_start_end=True
        )
    else:
        ds = Dataset(
            split=SPLIT,
            base_model_name=BASE_MODEL_NAME,
            fps=args.fps
        )

    if args.rank is not None and args.world_size is not None:
        ds = torch.utils.data.Subset(
            ds,
            list(range(args.RANK, len(ds), args.WORLD_SIZE))
        )

    feature_dir = get_feature_dir_eventful(
        args.dataset,
        BASE_MODEL_NAME,
        args.fps,
        SPLIT,
        top_r=args.top_r
    )

    config = PREDEFINED_PATHS['train'][f'eventful/{args.dataset}']
    config['decision_hyperparam'] = args.top_r

    model = EventfulCLIP(**config)
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
    cnt = 0
    desc = 'Running Eventful'
    pbar = tqdm(total=len(dataloader), desc=desc)

    prev_frame_idx = -sys.maxsize - 1
    prev_video_id = -sys.maxsize - 1

    for batch_idx, (frame_idx, pixel_values, original_outputs) in enumerate(dataloader):
        pixel_values = pixel_values.to(args.device).unsqueeze(1)
        original_outputs = original_outputs.to(args.device).unsqueeze(1)

        video_id = ds.id_idx[batch_idx][0]

        is_new_video = False

        if prev_frame_idx + 1 != frame_idx and prev_video_id != video_id:
            is_new_video = True

        # if is_new_video:
        #     print(f"######################################################################")

        if is_new_video:
            cached_states_for_next_batch_gate1 = None
            cached_states_for_next_batch_gate2 = None
            cached_states_for_next_batch_gate3 = None

        with torch.no_grad():
            (
                outputs,
                maps,
                cached_states_for_next_batch_gate1,
                cached_states_for_next_batch_gate2,
                cached_states_for_next_batch_gate3
            ) = model(
                pixel_values=pixel_values,
                cached_states_from_prev_batch_gate1=cached_states_for_next_batch_gate1,
                cached_states_from_prev_batch_gate2=cached_states_for_next_batch_gate2,
                cached_states_from_prev_batch_gate3=cached_states_for_next_batch_gate3,
            )

            # cached_states_for_next_batch_0123 = []
            # for cached_states in cached_states_for_next_batch_gate1:
            #     if cached_states is not None:
            #         cached_states = [c[:, -197:] for c in cached_states]
            #     cached_states_for_next_batch_0123.append(cached_states)

            # Shape: B, F
            sim = cosine_sim(outputs, original_outputs)
            # Shape: B, F
            mean_reuse_rates = maps.mean(dim=(2, 3))

            # sim_per_frame = sim.sum().item()
            # reuse_rate_per_frame = mean_reuse_rates.sum().item()

            if not is_new_video:
                total_sim += sim.sum().item()
                total_reuse_rate += mean_reuse_rates.sum().item()
                cnt += sim.numel()

            o_path = get_feature_path(feature_dir, video_id, 'o', frame_idx.item())
            o_embedding = outputs[0, 0]
            if not args.dry_run:
                o_path.parent.mkdir(exist_ok=True, parents=True)
                save_embedding(o_embedding, o_path)
            elif args.verbose >= 2:
                print(f'DRYRUN: Would save tensor of size {o_embedding.shape} to {o_path}')

        if cnt > 0:
            desc = f'Avg. sim: {total_sim / cnt:.2f}, Avg. reuse rate: {total_reuse_rate / cnt:2.2%}'
            pbar.set_description(desc)
        pbar.update()

        # print(f"v_id: {str(video_id).split('video')[-1]}, "
        #       f"f_id: {frame_idx.cpu().item():3d}, "
        #       f"sim: {sim_per_frame:1.2f}, "
        #       f"reuse_rate: {reuse_rate_per_frame*100:3.1f}%")

        prev_frame_idx = frame_idx
        prev_video_id = video_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['how2qa', 'msrvtt'], required=True)
    parser.add_argument('--fps', type=int, default=None, help='Override FPS instead of default')
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true', help='Do not save features')
    parser.add_argument('--debug-no-reuse', action='store_true', help='Disable reuse for debugging')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity level. Use -v for basic verbosity and -vv for more detailed."
    )
    parser.add_argument(
        '--top-r', type=int, default=5,
        help='Threshold for CMC model, note that threshold will be negated for implementation'
    )

    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=None)

    args = parser.parse_args()

    main(args)