## Reference: https://github.com/davidtvs/pytorch-lr-finder/blob/14abc0b8c3edd95eefa385c2619028e73831622a/torch_lr_finder/lr_finder.py

from matplotlib import pyplot as plt
import wandb
from torch.optim.lr_scheduler import _LRScheduler
from ...utils.train import reuse_loss

device = 'cuda'


class LRFinder(object):
    def __init__(
            self,
            model,
            optimizer,
            device=None,
            args=None,
            memory_cache=True,
            cache_dir=None
        ):
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self.model = model
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.device = device    
        self.args = args
    
    def range_test(
        self,
        train_loader,
        target=[True, True, True],
        val_loader=None,
        start_lr=None,
        end_lr=10,
        num_iter=100,
        smooth_f=0.05,
        diverge_th=8,
        accumulation_steps=1,
        logwandb=False
    ):
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Initialize the proper learning rate policy
        lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter, target)
        
        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        # Create an iterator to get data batch by batch
        iter_wrapper = DataLoaderIterWrapper(train_loader)

        first_target_idx = 0
        for t in target:
            if t:
                break
            first_target_idx += 1
        
        for iteration in range(num_iter):
            # Train on batch and retrieve loss
            loss = self._train_on_batch(iter_wrapper, accumulation_steps)
    
            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[first_target_idx])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss
                    
            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            
            if logwandb:
              wandb.log({'lr': lr_schedule.get_lr()[first_target_idx], 'loss': loss})

            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished")

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _train_on_batch(self, iter_wrapper, accumulation_steps):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            inputs, labels, compressed = iter_wrapper.get_batch()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            compressed = compressed.to(self.device)

            # Forward pass
            outputs, reuse_rate = self.model(inputs, compressed=compressed)
            args = self.args
            loss, *_ = reuse_loss(
                output=outputs,
                original_output=labels,
                reuse_maps=reuse_rate,
                target_reuse_rate=args.target_reuse_rate,
                target_similarity=args.target_similarity,
                sloss_scaler=args.sloss_scaler,
                rloss_scaler=args.rloss_scaler,
                use_cos_sim_loss=args.use_cos_sim_loss,
            )

            # Loss should be averaged in each step
            loss /= accumulation_steps

            loss.backward()

            if total_loss is None:
                total_loss = loss.item()
            else:
                total_loss += loss.item()

        self.optimizer.step()

        return total_loss

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")

        if show_lr is not None:
            plt.axvline(x=show_lr, color="red")
        plt.show()

    def get_best_lr(self):
      lrs = self.history['lr']
      losses = self.history['loss']
      return lrs[losses.index(min(losses))]

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, target, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.target = target
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        assert len(self.base_lrs) == len(self.target)
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        lrs = []
        for t, base_lr in zip(self.target, self.base_lrs):
            if t:
                lrs.append(base_lr * (self.end_lr / base_lr) ** r)
            else:
                lrs.append(base_lr)
        return lrs

class DataLoaderIterWrapper(object):
    def __init__(self, data_loader, auto_reset=True):
        self.data_loader = data_loader
        self.auto_reset = auto_reset
        self._iterator = iter(data_loader)

    def __next__(self):
        # Get a new set of inputs and labels
        try:
            batch = next(self._iterator)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)

        return batch

    def get_batch(self):
        return next(self)

