import torch
import torch.distributed
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np

from .reuse.model import ReuseModel
from .diffrate.model import DiffrateModel
from .dataset import How2qaTrainDataset, MsrvttTrainDataset
from ..utils import rename_base_model, PREDEFINED_PATHS, load_config, get_args
from ..utils.train import reuse_loss, reuse_loss_v2

from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size

from torch.distributed.elastic.multiprocessing.errors import record
import wandb
import os

def unpack_batch(batch, config, move_to_cuda=False):
    frame_idxs = batch[0]
    pixel_values = batch[1]
    original_hidden_states = batch[2]
    compressed = None
    ref_type = None
    ref_mask = batch[-1]


    optional_idx = 3
    original_output = batch[optional_idx]
    optional_idx += 1

    if config.use_compressed_info:
        compressed = batch[optional_idx]
        optional_idx += 1
    if config.decision_reference_type:
        ref_type = batch[optional_idx]
        optional_idx += 1

    ret = list((frame_idxs, pixel_values, original_hidden_states, original_output, compressed, ref_type, ref_mask))
    if move_to_cuda:
        for idx, t in enumerate(ret):
            if t is not None:
                ret[idx] = t.to('cuda')
    return ret

def is_main_process(accelerator=None):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 and torch.distributed.get_rank() != 0:
        return False
    if accelerator is None:
        return True
    return accelerator.is_local_main_process

def save_model(accelerator, model, config, checkpoint_path):
    assert accelerator is not None, 'We do not support saving model without accelerator at the moment'
    if not is_main_process(accelerator):
        return

    if torch.distributed.is_initialized():
        model = model.module

    if config.use_lora:
        model.model.merge_adapter()

    accelerator.save_model(model, str(checkpoint_path))

    if config.use_lora:
        model.model.unmerge_adapter()

@record
def training_function(config):
    # Setup project names
    base_model_renamed = rename_base_model(config['base_model_name'])
    if config.debug:
        project_dir = Path('/tmp') / base_model_renamed
    else:
        # project_dir = Path('/mnt/nfs') / base_model_renamed
        project_dir = Path(PREDEFINED_PATHS["train"]["checkpoint_dir"]) / base_model_renamed
    # Generate unique identifier based on current timestamp
    if config.debug:
        accelerator = None
    else:
        accelerator = Accelerator(
            log_with='wandb',
            project_dir=str(project_dir),
            step_scheduler_with_optimizer=False,
        )

    wandb_run_id = os.environ.get('WANDB_RUN_ID', None)
    if wandb_run_id is not None:
        config.try_name = wandb_run_id

    config.checkpoint_dir = project_dir / config.name.replace('-', '/') / config.try_name / 'checkpoints'
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @find_executable_batch_size(starting_batch_size=config.batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator
        if accelerator is not None:
            accelerator.free_memory()

        # Initialize Model
        if config.diffrate_model_name is not None:
            model = DiffrateModel(
                base_model_name=config.base_model_name,
                dataset=config.dataset,
                diffrate_model_name=config.diffrate_model_name,
                use_lora=config.use_lora,
                lora_targets=config.lora_targets,
                lora_rank = config.lora_rank,
            )
        else:
            model = ReuseModel(
                **config,
            )

        ds_kwargs = {}
        # Load Dataset
        if config.dataset == 'how2qa':
            TrainDataset = How2qaTrainDataset
            TestDataset = How2qaTrainDataset
            train_split = 'test'
            test_split = 'frozenbilm'
            train_use_start_end = config.use_start_end
            test_use_start_end = config.use_start_end
        elif config.dataset == 'msrvtt':
            TrainDataset = MsrvttTrainDataset
            TestDataset = MsrvttTrainDataset
            train_split = 'train'
            test_split = 'test'
            train_use_start_end = config.use_start_end
            test_use_start_end = config.use_start_end
        else:
            raise NotImplementedError

        train_dataset = TrainDataset(
            pattern=config.frame_stack_pattern,
            split=train_split,
            base_model_name=config.base_model_name,
            fps=config.fps,
            return_hidden_states=True,
            return_compressed=config.use_compressed_info,
            step=config.train_dataset_step,
            use_coded_order=config.use_coded_order,
            augment_far_ratio=config.augment_far_ratio,
            augment_same_ratio=config.augment_same_ratio,
            augment_short_ratio=config.augment_short_ratio,
            augment_short_far_ratio=config.augment_short_far_ratio,
            return_separate_ref_type=config.decision_reference_type,
            use_start_end=train_use_start_end,
            **ds_kwargs,
        )

        test_dataset = TestDataset(
            pattern=config.test_frame_stack_pattern,
            split=test_split,
            base_model_name=config.base_model_name,
            fps=config.fps,
            return_hidden_states=True,
            return_compressed=config.use_compressed_info,
            step=config.train_dataset_step,
            return_separate_ref_type=config.decision_reference_type,
            use_start_end=test_use_start_end,
            **ds_kwargs,
        )

        if config.train_sample_rate is not None:
            num_samples = int(float(config.train_sample_rate) * len(train_dataset))
            train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        if config.test_sample_rate is not None:
            num_samples = int(float(config.test_sample_rate) * len(test_dataset))
            test_sampler = RandomSampler(test_dataset, replacement=True, num_samples=num_samples)
        else:
            test_sampler = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_worker,
            sampler=train_sampler
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_worker,
            sampler=test_sampler
        )

        codecnet_params = [p for n, p in model.named_parameters() if 'codecnet' in n]
        restoration_params = [p for n, p in model.named_parameters() if 'restoration' in n]
        decision_params = [p for n, p in model.named_parameters() if 'decision' in n]
        finetune_params = []

        for n, p in model.named_parameters():
            if 'codecnet' not in n and 'decision' not in n and 'restoration' not in n:
                if p.requires_grad:
                    finetune_params.append(p)
        print(f"Fine-tuning {len(finetune_params)} parameters")

        if config.finetune_only:
            for p in codecnet_params:
                p.requires_grad = False
            for p in decision_params:
                p.requires_grad = False

        optimizer = AdamW(
            [
                {"params": codecnet_params, "lr": config.codecnet_lr},
                {"params": restoration_params, "lr": config.restoration_lr},
                {"params": finetune_params, "lr": config.finetune_lr},
                {"params": decision_params, "lr": config.decision_lr},
            ],
        )
        if config.lr_scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=config.lr_patience
            )
        elif config.lr_scheduler == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.lr_exponential_gamma
            )
        elif config.lr_scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,
                eta_min=5e-6
            )
        else:
            raise NotImplementedError

        # Training Loop
        early_stop_counter = 0
        best_loss = float('inf')
        recent_best_loss = float('inf')

        start_epoch = 0  # by default, start from scratch

        if accelerator is not None:
            model= accelerator.prepare(
                model,
                device_placement=([True])
            )
            if config.resume_reuse_name is not None:
                resume_checkpoints = project_dir / config.name.replace('-', '/') / config.resume_reuse_name / 'checkpoints' / 'best'
                accelerator.load_state(resume_checkpoints, strict=False)

            optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
                optimizer, train_dataloader, test_dataloader, lr_scheduler,
                device_placement=(True, True, True, True)
            )
            accelerator.register_for_checkpointing(lr_scheduler)
        else:
            model = model.to('cuda')

        rank = int(os.environ.get('RANK', 0))
        if accelerator is not None and accelerator.is_local_main_process and rank == 0:

            accelerator.init_trackers(
                project_name=config.name,
                config=config.to_dict(),
                # init_kwargs={"wandb": {"name": config.try_name}},
            )
            wandb.watch(model, log='all', log_freq=100)

        tqdm_kwargs = {}
        if config.gating_scheduling:
            tau_initial = 1.
            tau_final = 0.01
            T = 30
        tau = None
        for epoch in range(start_epoch, config.epochs):
            # model.train()
            loss_batch = 0

            if config.gating_scheduling:
                tau = tau_initial * (tau_final / tau_initial) ** (max(epoch, T) / T)
            # tau = 0.67 # Adafuse

            model.train()  # Set model to training mode
            for batch in tqdm(
                train_dataloader,
                disable=accelerator is not None and not accelerator.is_local_main_process,
                **tqdm_kwargs
                ):
                (
                    frame_idxs,
                    pixel_values,
                    original_hidden_states,
                    original_output,
                    compressed,
                    ref_type,
                    ref_mask
                ) = unpack_batch(batch, config, move_to_cuda=accelerator is None)
                output = model(
                    pixel_values,
                    compressed=compressed,
                    output_hidden_states=True,
                    tau=tau,
                    hard=config.gating_hard,
                    ref_mask=ref_mask,
                    ref_type=ref_type,
                )
                if config.diffrate_model_name is None:
                    output, reuse_maps, hidden_states = output
                else:
                    output = output.image_embeds
                    # FIXME: fix shape
                    reuse_maps = torch.zeros(1, device=output.device)

                # Compute Loss
                # loss, cos_error, mse_error, reuse_maps = reuse_loss(
                #     output=output,
                #     original_hidden_states=original_hidden_states,
                #     original_output=original_output,
                #     reuse_maps=reuse_maps,
                #     target_reuse_rate=config.target_reuse_rate,
                #     target_similarity=config.target_similarity,
                #     sloss_scaler=config.sloss_scaler,
                #     rloss_scaler=config.rloss_scaler,
                #     use_cos_sim_loss=config.use_cos_sim_loss,
                #     use_min_cos_sim_loss=config.use_min_cos_sim_loss,
                #     sloss_pattern=config.sloss_pattern,
                #     rloss_pattern=config.rloss_pattern,
                #     use_hidden_states=config.use_hidden_states,
                # )
                loss, hidden_error, cls_error, reuse_maps, reuse_rate_per_frame = reuse_loss_v2(
                    hidden_states=hidden_states,
                    output=output,
                    original_hidden_states=original_hidden_states,
                    original_output=original_output,
                    reuse_maps=reuse_maps,
                    target_reuse_rate=config.target_reuse_rate,
                    hloss_scaler=config.hloss_scaler,
                    sloss_scaler=config.sloss_scaler,
                    rloss_scaler=config.rloss_scaler,
                    sloss_pattern=config.sloss_pattern,
                    rloss_pattern=config.rloss_pattern,
                    rloss_duplicate_final_frame=config.rloss_duplicate_final_frame,
                )

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

                # Optimize
                optimizer.step()
                optimizer.zero_grad()

                metrics = {
                    'train/loss': loss.item(),
                }
                metrics['train/hidden_err'] = hidden_error.item()
                metrics['train/cls_err'] = cls_error.item()
                metrics['train/reuse_rate'] = reuse_maps.item()

                reuse_rate_per_frame = reuse_rate_per_frame.mean(dim=0)
                for r_idx, r in enumerate(reuse_rate_per_frame):
                    metrics[f'train/reuse_rate_{r_idx}'] = r.item()

                # Log Metrics
                if accelerator is not None:
                    accelerator.log(
                        metrics,
                    )

                # Sum batch loss
                loss_batch += loss.item()

            loss_batch /= len(train_dataloader)
            # Print Epoch Summary
            if accelerator is None or accelerator.is_local_main_process:
                print(f"Epoch {epoch+1} loss: {loss_batch}")

            if accelerator is not None:
                accelerator.log({'train/loss_epoch': loss_batch})

            val_reuse_rates = []
            # Validation Phase
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No need to calculate gradients
                val_loss_batch = 0
                val_cls_error_batch = 0
                val_hidden_error_batch = 0
                for batch in tqdm(
                    test_dataloader,
                    disable=accelerator is not None and not accelerator.is_local_main_process,
                    **tqdm_kwargs
                ):
                    # Unpack batch
                    (
                        frame_idxs,
                        pixel_values,
                        original_hidden_states,
                        original_output,
                        compressed,
                        ref_type,
                        ref_mask
                    ) = unpack_batch(batch, config, move_to_cuda=accelerator is None)

                    # Forward Pass
                    output = model(
                        pixel_values,
                        compressed=compressed,
                        output_hidden_states=True,
                        tau=tau,
                        hard=True,
                        ref_mask=ref_mask,
                        ref_type=ref_type,
                    )
                    if config.diffrate_model_name is None:
                        output, reuse_maps, hidden_states = output
                    else:
                        output = output.image_embeds
                        reuse_maps = torch.ones(1, device=output.device)

                    if (
                        accelerator is not None 
                        and torch.distributed.is_initialized()
                        and torch.distributed.get_world_size() > 1
                    ):
                        if config.diffrate_model_name is None:
                            (
                                all_output,
                                all_reuse_maps,
                                all_hidden_states,
                                all_original_hidden_states,
                                all_original_output
                            ) = accelerator.gather_for_metrics((
                                output, reuse_maps, hidden_states, original_hidden_states, original_output
                            ))
                        else:
                            all_output, all_original_output = accelerator.gather_for_metrics((output, original_output))
                    else:
                        all_hidden_states = hidden_states
                        all_output = output
                        all_original_output = original_output
                        all_original_hidden_states = original_hidden_states
                        all_reuse_maps = reuse_maps

                    val_loss, hidden_error, cls_error, reuse_maps, reuse_rate_per_frame = reuse_loss_v2(
                        hidden_states=all_hidden_states,
                        output=all_output,
                        original_hidden_states=all_original_hidden_states,
                        original_output=all_original_output,
                        reuse_maps=all_reuse_maps,
                        target_reuse_rate=config.target_reuse_rate,
                        sloss_scaler=config.sloss_scaler,
                        rloss_scaler=config.rloss_scaler,
                        sloss_pattern=config.sloss_pattern,
                        rloss_pattern=config.rloss_pattern,
                        max_reuse_per_layer=config.max_reuse_per_layer,
                        rloss_duplicate_final_frame=config.rloss_duplicate_final_frame,
                    )

                    # Sum batch loss
                    val_loss_batch += val_loss.item()
                    if config.diffrate_model_name is None:
                        val_hidden_error_batch += hidden_error.item()
                        val_cls_error_batch += cls_error.item()
                        val_reuse_rates.append(reuse_rate_per_frame.mean(dim=0))

            val_loss_batch /= len(test_dataloader)
            if config.diffrate_model_name is None:
                val_hidden_error_batch /= len(test_dataloader)
                val_cls_error_batch /= len(test_dataloader)

                val_reuse_rates = torch.stack(val_reuse_rates, dim=0)
                val_reuse_rate_per_frame_mean = val_reuse_rates.mean(dim=0)
                val_reuse_rate_per_frame_std = val_reuse_rates.std(dim=0)
                val_reuse_rate_batch = torch.mean(val_reuse_rate_per_frame_mean)

            # log learning rate & update
            current_lr = optimizer.param_groups[0]['lr']
            metrics = {
                'val/loss': val_loss_batch,
                'train/lr': current_lr,
                'val/epoch': epoch,
            }


            if config.diffrate_model_name is None:
                metrics['val/hidden_err'] = val_hidden_error_batch
                metrics['val/cls_err'] = val_cls_error_batch
                metrics['val/reuse_rate'] = val_reuse_rate_batch

                for frame_type_idx in range(val_reuse_rate_per_frame_mean.shape[0]):
                    metrics[f'val/reuse_rate_{frame_type_idx}'] = val_reuse_rate_per_frame_mean[frame_type_idx].item()
                    metrics[f'val/reuse_rate_{frame_type_idx}_std'] = val_reuse_rate_per_frame_std[frame_type_idx].item()


            if accelerator is not None and accelerator.is_local_main_process:
                print(f"Epoch {epoch} Validation loss: {val_loss_batch}")
                for k, v in metrics.items():
                    print(f"VALIDATION {epoch} {k} {v}")
                accelerator.log(
                    metrics,
                )

            if config.lr_scheduler == 'plateau':
                lr_scheduler.step(val_loss_batch)
            elif config.lr_scheduler == 'exponential':
                lr_scheduler.step()
            elif config.lr_scheduler == 'cosine':
                lr_scheduler.step()
            else:
                raise NotImplementedError

            if val_loss_batch < best_loss:
                best_loss = val_loss_batch
                checkpoint_path = Path(config.checkpoint_dir) / "best"
                if accelerator is not None:
                    save_model(accelerator, model, config, checkpoint_path)
                early_stop_counter = 0  # Reset counter

            if val_loss_batch < recent_best_loss:
                recent_best_loss = val_loss_batch
                early_stop_counter = 0  # Reset counter
            else:
                early_stop_counter += 1

            # If performance hasn't improved for 'early_stop_patience' epochs, stop training
            if early_stop_counter >= config.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

            if not config.is_sweep and not config.debug:
                if is_main_process(accelerator):
                    checkpoint_dir = Path(config.checkpoint_dir) / f'epoch_{epoch}'
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_model(accelerator, model, config, checkpoint_dir)

                    aux_path = checkpoint_dir / f"aux.bin"
                    aux = {
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'recent_best_loss': recent_best_loss,
                    }
                    torch.save(aux, aux_path)


    inner_training_loop()
    if accelerator is not None:
        accelerator.end_training()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('CONFIG_YAML_PATH', type=str)
    config = parser.parse_args()

    config = load_config(config.CONFIG_YAML_PATH)

    # Set random seed
    torch.manual_seed(config['seed'])
    training_function(config)

