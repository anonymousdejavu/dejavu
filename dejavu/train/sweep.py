import torch

from ..utils import load_config
from .__main__ import training_function

from torch.distributed.elastic.multiprocessing.errors import record
import wandb

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('CONFIG_YAML_PATH', type=str)
    config = parser.parse_args()

    config = load_config(config.CONFIG_YAML_PATH)

    # For sweeping how2qa
    config['is_sweep'] = True
    config['train_sample_rate'] = 0.01
    config['test_sample_rate'] = 0.01
    # config['epoch'] = 50

    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val/loss"},
        "parameters": {
            "decision_lr": {
                "distribution": "log_uniform_values",
                "max": 1e-2,
                "min": 1e-6,
            },
            "codecnet_lr": {
                "distribution": "log_uniform_values",
                "max": 1e-2,
                "min": 1e-6,
            },
            "restoration_lr": {
                "distribution": "log_uniform_values",
                "max": 1e-2,
                "min": 1e-6,
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "s": 2,
            "eta": 3,
            "max_iter": 27,
        },
    }

    # Set random seed
    torch.manual_seed(config['seed'])

    sweep_id = wandb.sweep(sweep_config, project=config['name'])

    wandb.agent(sweep_id, lambda: training_function(config), count=500)