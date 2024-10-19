import torch
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from torch.optim import AdamW
import argparse

from ..fixed_pattern.model import ReuseModel as FixedReuseModel
from ..diffrate.model import DiffrateModel
from ...utils import rename_base_model, parse_lora_targets
from ...utils import PREDEFINED_PATHS
from ...utils.train import load_dataset, get_args

from torch.distributed.elastic.multiprocessing.errors import record
import wandb

from .lr_finder import LRFinder

@record
def training_function(args):
    # Setup project names
    base_model_renamed = rename_base_model(args.base_model_name)

    def inner_training_loop(batch_size):
        # Initialize Model
        if args.use_fixed_pattern and args.ref_type=="p4_p2n2_p1n1":
            assert args.ref_space == "global"
            dataset = None 
            if args.dataset == "msrvtt": dataset = "msrvtt"
            model = FixedReuseModel(
                **vars(args),
            )

        elif args.diffrate_model_name is not None:
            model = DiffrateModel(
                base_model_name=args.base_model_name,
                dataset=args.dataset,
                diffrate_model_name=args.diffrate_model_name,
                use_lora=args.use_lora,
                lora_targets=parse_lora_targets(args.lora_targets),
                lora_rank = args.lora_rank,
            )
        else:
            raise NotImplementedError

        train_dataset, test_dataset = load_dataset(args)

        if args.sample_rate is not None:
            num_samples = int(float(args.sample_rate) * len(train_dataset))
            train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
            num_samples = int(float(args.sample_rate) * len(test_dataset))
            test_sampler = RandomSampler(test_dataset, replacement=True, num_samples=num_samples)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

            num_samples = int(0.1 * len(test_dataset))
            test_sampler = RandomSampler(test_dataset, replacement=True, num_samples=num_samples)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_worker,
            sampler=train_sampler
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_worker,
            sampler=test_sampler
        )

        codecnet_params = [p for n, p in model.named_parameters() if 'codecnet' in n]
        decision_params = [p for n, p in model.named_parameters() if 'decision' in n]
        finetune_params = []

        for n, p in model.named_parameters():
            if 'codecnet' not in n and 'decision' not in n:
                finetune_params.append(p)

        target = [True, False, True]
        optimizer = AdamW(
            [
                {"params": codecnet_params, "lr": 1e-6}, # 0.01
                {"params": finetune_params, "lr": 0.075}, # 0.01
                {"params": decision_params, "lr": 1e-6}, # 0.2
            ],
        )

        model = model.to('cuda')

        # Initialize wandb and LR Finder
        wandb.init(project='lrfinder')
        wandb.watch(model, log='all')
        lr_finder = LRFinder(model, optimizer, device='cuda', args=args)
        lr_finder.range_test(
            train_dataloader,
            target=target,
            end_lr=10,
            num_iter=100,
            accumulation_steps=16,
            logwandb=True
        )

    inner_training_loop(args.batch_size)


if __name__ == '__main__':
    args = get_args()

    # Set random seed
    torch.manual_seed(args.seed)
    training_function(args)
