#!/usr/bin/env python3

import builtins
import numpy as np
import os
import sys
import torch
import wandb

from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn


def suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=" ", _end="\n", _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def suppress_wandb():
    """Suppresses wandb logging from the current_process."""
    def ignore(data, step=None, commit=None, sync=None):
        pass
    wandb.log = ignore


def dump_cfg(cfg, out_dir):
    """Dumps a config to dir."""
    out_f = os.path.join(out_dir, "config.yaml")
    with open(out_f, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print("Wrote config to: {}".format(out_f))


def setup_env(cfg):
    """Sets up environment for training or testing."""
    if cfg.num_gpus > 1:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0
    if rank == 0:
        exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
        checkpoint_dir = f"{exp_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        dump_cfg(cfg, exp_dir)
        # Automatically increments run index.
        wandb_runs = wandb.Api(timeout=19).runs(
            path=f"{cfg.wandb.entity}/{cfg.wandb.project}", order="-created_at"
        )
        try:
            prev_name = wandb_runs[0].name
            run_index = int(prev_name.split("_")[0]) + 1
        except:
            run_index = 0
        name = f"{run_index:04d}_{cfg.wandb.name}"
        wandb.init(
            name=name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg),
            config_exclude_keys=["wandb", "hydra"],
        )
    else:
        suppress_print()
        suppress_wandb()
    seed = cfg.rng_seed * world_size + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    return seed


def construct_loader(dataset, mb_size, num_gpus, shuffle, drop_last, num_workers):
    """Constructs a data loader."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(mb_size / num_gpus),
        shuffle=(False if num_gpus > 1 else shuffle),
        sampler=(DistributedSampler(dataset, shuffle=shuffle) if num_gpus > 1 else None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True
    )


def shuffle(loader, cur_epoch):
    """Shuffles the data."""
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler))
    # RandomSampler handles shuffling automatically
    # DistributedSampler shuffles data based on epoch
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)


def lr_fun_cos(cfg, cur_epoch):
    """Computes lr according to cosine lr schedule."""
    base_lr, num_ep = cfg.train.base_lr, cfg.train.num_ep
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / num_ep))


def lr_fun_fixed(cfg, cur_epoch):
    """Compute lr according to fixed lr schedule."""
    return cfg.train.base_lr


def update_lr(cfg, optimizer, cur_epoch):
    """Updates lr for current epoch."""
    lr_fun = globals()["lr_fun_" + cfg.train.lr_sch]
    new_lr = lr_fun(cfg, cur_epoch)
    if cur_epoch < cfg.train.warmup_epochs:
        alpha = cur_epoch / cfg.train.warmup_epochs
        warmup_factor = cfg.train.warmup_factor * (1.0 - alpha) + alpha
        new_lr *= warmup_factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


@torch.no_grad()
def spread_losses(losses, steps, minimize=True):
    # This function samples evenly-spaced losses (as evenly-spaced as possible) from a run.
    # losses should be size (N, num_losses)
    assert losses.dim() == 2
    N, num_losses = losses.size()
    min_loss = losses[:, 0]
    if minimize:
        max_loss = losses.amin(dim=1)
    else:
        max_loss = losses.amax(dim=1)
    ideal = torch.stack([torch.linspace(min_loss[i].item(), max_loss[i].item(), steps) for i in range(N)])
    dists = (ideal.view(N, steps, 1) - losses.view(N, 1, num_losses)).abs()
    nearest_neighbors = dists.argmin(dim=2)
    spread_out = losses.gather(dim=1, index=nearest_neighbors)
    assert spread_out.shape == (N, steps)
    return spread_out


def accumulate(model1, model2, decay=0.9999):  # https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    alpha = 1 - decay

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=alpha)


def requires_grad(model, flag=False):   # https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
    for p in model.parameters():
        p.requires_grad = flag

