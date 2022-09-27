#!/usr/bin/env python3

import argparse
import datetime
import decimal
import itertools
import json
import math
import numpy as np
import os
import pickle
import simplejson
import time

from collections import deque
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets

from Gpt.data.database import Database
from Gpt.data.augment import permute_in, permute_out, permute_in_out


def random_permute_cnn(net, generator=None, permute_in_fn=permute_in, permute_out_fn=permute_out,
                       permute_in_out_fn=permute_in_out, register_fn=lambda x: x):
    # NOTE: when using this function as part of random_permute_flat, THE ORDER IN WHICH
    # PERMUTE_OUT_FN, PERMUTE_IN_FN, etc. get called IS REALLY IMPORTANT. The order MUST be consistent
    # with whatever net.state_dict().keys() returns, otherwise the permutation will be INCORRECT.
    # If you're using this function directly on a ConvNet instance and then flattening its weights,
    # the order does NOT matter since everything is being done in-place.
    assert isinstance(net, ConvNet)
    running_permute = None  # Will be set by initial nn.Linear
    convs = [module for module in net.modules() if isinstance(module, nn.Conv2d)]
    for ix, conv in enumerate(convs):
        if ix == 0:  # Input layer
            running_permute = torch.randperm(conv.weight.size(0), generator=generator)
            permute_out_fn(conv.weight, running_permute)
        else:  # All other conv layers:
            new_permute = torch.randperm(conv.weight.size(0), generator=generator)
            permute_in_out_fn(conv.weight, running_permute, new_permute)
            running_permute = new_permute
    # Handle final linear layer to logits:
    permute_in_fn(net.fc.weight, running_permute)
    register_fn(net.fc.bias)

###############################################################################
# Model
###############################################################################

class ConvNet(nn.Module):

    def __init__(self, ws=[16, 32]):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, ws[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ws[0], ws[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ws[1], 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


###############################################################################
# Dataset
###############################################################################

def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode='constant')


def horizontal_flip(im, p):
    """Performs horizontal flip (CHW format)."""
    flip = np.random.uniform() < p
    if flip:
        im = im[:, :, ::-1]
    return im


def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y:(y + size), x:(x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop


# Data path
_DATA_PATH = "vision_datasets"

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, split):
        assert split in ['train', 'test']
        print('Constructing CIFAR-10 {}...'.format(split))
        # Download CIFAR-10 if needed (we use a separate dataset class):
        torchvision.datasets.CIFAR10(root=_DATA_PATH, download=True)
        root = f"{_DATA_PATH}/cifar-10-batches-py"
        self._data_path = root
        self._split = split
        # Data format:
        #   self._inputs - (split_size, 3, 32, 32) ndarray
        #   self._labels - split_size list
        self._inputs, self._labels = self._load_data()

    def _load_batch(self, batch_path):
        with open(batch_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], d[b"labels"]

    def _load_data(self):
        """Loads data in memory."""
        print("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == 'train':
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            inputs_batch, labels_batch = self._load_batch(batch_path)
            inputs.append(inputs_batch)
            labels += labels_batch
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, 32, 32))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = color_norm(im, _MEAN, _SD)
        if self._split == 'train':
            im = horizontal_flip(im=im, p=0.5)
            im = random_crop(im=im, size=32, pad_size=4)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]


###############################################################################
# Loader
###############################################################################

def construct_loader(dataset, mb_size, shuffle, drop_last):
    """Constructs a data loader."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=mb_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True
    )


###############################################################################
# Meters
###############################################################################

def log_json_stats(stats):
    """Logs json stats."""
    stats = {
        k: decimal.Decimal('{:.6f}'.format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    print(json_stats)


def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{0:02},{1:02}:{2:02}:{3:02}'.format(days, hrs, mins, secs)


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, max_epoch, log_period=10):
        self.epoch_iters = epoch_iters
        self.max_epoch = max_epoch
        self.log_period = log_period
        self.max_iter = max_epoch * epoch_iters
        self.iter_timer = Timer()
        # Train minibatch stats (tracked over a window)
        self.train_loss = ScalarMeter(log_period)
        self.train_err = ScalarMeter(log_period)
        # Test set stats (tracked over a window)
        self.test_loss = ScalarMeter(log_period)
        self.test_err = ScalarMeter(log_period)

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.train_loss.reset()
        self.train_err.reset()
        self.test_loss.reset()
        self.test_err.reset()

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, train_loss, train_err, test_loss, test_err):
        # Train minibatch stats
        self.train_loss.add_value(train_loss)
        self.train_err.add_value(train_err)
        # Test set stats
        self.test_loss.add_value(test_loss)
        self.test_err.add_value(test_err)

    def get_iter_stats(self, cur_epoch, cur_iter):
        iters = cur_epoch * self.epoch_iters + cur_iter + 1
        eta_sec = self.iter_timer.average_time * (self.max_iter - iters)
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        stats = {
            '_type': 'train_iter',
            'epoch': '{}/{}'.format(cur_epoch + 1, self.max_epoch),
            'iter': '{}/{}'.format(cur_iter + 1, self.epoch_iters),
            'time_avg': self.iter_timer.average_time,
            'eta': eta_str(eta_td),
            'train_loss': self.train_loss.get_win_median(),
            'train_err': self.train_err.get_win_median(),
            'test_loss': self.test_loss.get_win_median(),
            'test_err': self.test_err.get_win_median(),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % self.log_period != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        iters = (cur_epoch + 1) * self.epoch_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - iters)
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        stats = {
            '_type': 'train_epoch',
            'epoch': '{}/{}'.format(cur_epoch + 1, self.max_epoch),
            'time_avg': self.iter_timer.average_time,
            'eta': eta_str(eta_td),
            'test_loss': self.test_loss.deque[-1],
            'test_err': self.test_err.deque[-1],
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        log_json_stats(stats)


###############################################################################
# Metrics
###############################################################################

def top1_error(preds, labels):
    """Computes the top-1 error."""
    max_inds = preds.argmax(dim=1)
    inst_top1_correct = max_inds.eq(labels)
    inst_top1_err = (1.0 - inst_top1_correct.float()) * 100.0
    top1_err = inst_top1_err.mean().item()
    return top1_err, inst_top1_err


def params_count(model):
    """Computes the number of parameters."""
    return np.sum([p.numel() for p in model.parameters()]).item()


###############################################################################
# Training
###############################################################################

def log_model_info(model):
    """Logs model info."""
    print("Model:\n{}".format(model))
    print("Params: {:,}".format(params_count(model)))


def get_param_sizes(state_dict):
    return torch.tensor([p.numel() for p in state_dict.values()], dtype=torch.long)


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters)


def save_checkpoint_lmdb(
    cfg, database, runs, model, cur_epoch, cur_iter,
    avg_train_loss, avg_train_err, avg_test_loss, avg_test_err
):
    dirname = os.path.basename(cfg.job_dir)
    key = "{}/ep{:03d}_it{:05d}.pt".format(dirname, cur_epoch, cur_iter)
    state_dict = model.state_dict()
    flat_params = get_flat_params(state_dict)
    database[key] = flat_params.cpu()
    database[f"{key}_arch"] = get_param_sizes(state_dict)
    runs["checkpoints"][key] = {
        "avg_train_loss": avg_train_loss,
        "avg_train_err": avg_train_err,
        "avg_test_loss": avg_test_loss,
        "avg_test_err": avg_test_err
    }
    def update_min_max(kvs, k, v, op):
        kvs[k] = op(kvs[k], v)
    runs["metadata"]["avg_test_loss"].append(avg_test_loss)
    runs["metadata"]["avg_test_err"].append(avg_test_err)
    update_min_max(runs["metadata"], "optimal_test_loss", avg_test_loss, min)
    update_min_max(runs["metadata"], "optimal_test_err", avg_test_err, min)
    update_min_max(runs["metadata"], "min_parameter_val", flat_params.min().item(), min)
    update_min_max(runs["metadata"], "max_parameter_val", flat_params.max().item(), max)


def lr_fun_cos(base_lr, cur_epoch, max_epoch):
    """Cosine learning rate schedule."""
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))


def update_lr(optimizer, base_lr, cur_epoch, max_epoch):
    """Updates lr for current epoch."""
    new_lr = lr_fun_cos(base_lr, cur_epoch, max_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


@torch.inference_mode()
def test_epoch(inputs, labels, model):
    """Evaluates the model on the test set."""
    model.eval()
    preds = model(inputs)
    inst_loss = F.cross_entropy(preds, labels, reduction='none')
    test_loss = inst_loss.mean().item()
    top1_err, inst_top1_err = top1_error(preds, labels)
    return test_loss, top1_err, inst_loss, inst_top1_err


def train_epoch(
    cfg, database, runs, train_loader, test_inputs, test_labels, model, optimizer,
    train_meter, cur_epoch, epoch_iters, save_iters
):
    """Performs one epoch of training."""
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        inst_loss = F.cross_entropy(preds, labels, reduction='none')
        loss = inst_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, (err, inst_err) = loss.item(), top1_error(preds, labels)
        abs_iter = cur_epoch * epoch_iters + cur_iter + 1
        if abs_iter in save_iters:
            test_loss, test_err, _, _ = \
                test_epoch(test_inputs, test_labels, model)
            save_checkpoint_lmdb(
                cfg, database, runs, model, cur_epoch, cur_iter + 1,
                loss, err, test_loss, test_err
            )
        train_meter.iter_toc()
        train_meter.update_stats(loss, err, 0, 0)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        model.train()
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def unload_test_set():
    test_dataset = Cifar10("test")
    test_images, test_labels = next(
        iter(construct_loader(test_dataset, len(test_dataset), shuffle=False, drop_last=False)))
    test_images, test_labels = test_images.cuda(), test_labels.cuda(non_blocking=True)
    assert test_images.size(0) == test_labels.size(0) == len(test_dataset)
    return test_images, test_labels


def train(cfg):
    """Trains a model."""

    # Save checkpoints to lmdb
    database = Database(cfg.job_dir, readonly=False)

    # Compute checkpoints metadata
    runs = {
        "checkpoints": {},
        "metadata": {
            "avg_test_loss": [],
            "avg_test_err": [],
            "optimal_test_loss": float("inf"),
            "optimal_test_err": float("inf"),
            "min_parameter_val": float("inf"),
            "max_parameter_val": float("-inf")
        }
    }

    # Construct the model
    model = ConvNet(ws=[16, 32]).cuda()
    log_model_info(model)

    # Construct the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.base_lr, momentum=0.9, weight_decay=cfg.wd,
        dampening=0.0, nesterov=True
    )

    # Create datasets
    train_dataset = Cifar10("train")

    # Create data loaders
    train_loader = construct_loader(train_dataset, cfg.mb_size, shuffle=True, drop_last=True)
    # Move entirety of test dataset to GPU for fast evaluation:
    test_images, test_labels = unload_test_set()

    # Track training stats
    train_meter = TrainMeter(len(train_loader), cfg.max_epoch)

    # Save initial (randomly-initialized) network and corresponding data:
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
    preds = model(inputs)
    inst_loss = F.cross_entropy(preds, labels, reduction='none')
    loss, (err, inst_err) = inst_loss.mean().item(), top1_error(preds, labels)
    test_loss, test_err, _, _ = test_epoch(test_images, test_labels, model)
    print(f"Randomly-Initialized Test Loss: {test_loss}")
    save_checkpoint_lmdb(
        cfg, database, runs, model, 0, 0, loss, err, test_loss, test_err
    )

    # Choose checkpoints to save randomly (+ final checkpoint)
    epoch_iters = len(train_loader)
    total_iters = epoch_iters * cfg.max_epoch
    cand_iters = np.arange(1, total_iters)
    rand_iters = np.random.choice(cand_iters, size=(cfg.num_save - 2), replace=False)
    save_iters = set(rand_iters)
    save_iters.add(total_iters)

    # Perform the training loop
    for cur_epoch in range(cfg.max_epoch):
        update_lr(optimizer, cfg.base_lr, cur_epoch, cfg.max_epoch)
        train_epoch(
            cfg, database, runs, train_loader, test_images, test_labels, model, optimizer,
            train_meter, cur_epoch, epoch_iters, save_iters
        )

    # Save metadata
    runs_path = os.path.join(cfg.job_dir, "runs.json")
    with open(runs_path, "w") as f:
        json.dump(runs, f, indent=2, sort_keys=True)

    # Mark job as complete
    done_path = os.path.join(cfg.job_dir, "DONE.txt")
    try:
        os.mknod(done_path)
    except FileExistsError:
        collision_path = os.path.join(cfg.job_dir, "WARNING_COLLISION.txt")
        os.mknod(collision_path)


###############################################################################
# Running
###############################################################################

class AttrDict(dict):
    """Data structure for the config."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_rng_seed(seed):
    """Sets RNG seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def dump_cfg(cfg, out_dir):
    """Writes a config to dir."""
    return
    out_f = os.path.join(out_dir, "config.json")
    with open(out_f, mode="w") as f:
        json.dump(cfg, f)
    print("Wrote config to: {}".format(out_f))


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="new_checkpoint_data/cifar10")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    # Ensure the root out dir exists
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Construct the job out dir
    seed = len(list(glob(f"{out_dir}/*")))
    job_dir = f"{out_dir}/{seed:06d}"
    os.makedirs(job_dir, exist_ok=False)

    # Fix the RNG seed
    set_rng_seed(seed)
    print("Training with seed: {}".format(seed))

    # Construct the training config
    cfg = AttrDict({
        "base_lr": 0.1,
        "wd": 0.0005,
        "max_epoch": 50,
        "mb_size": 128,
        "job_dir": job_dir,
        "num_save": 200
    })
    dump_cfg(cfg, job_dir)

    # Train the model
    train(cfg)


if __name__ == "__main__":
    main()
