#!/usr/bin/env python3

"""Meters."""

import decimal
import numpy as np
import simplejson
import time
import torch
import wandb

from collections import deque


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


def float_to_decimal(data, prec=5):
    """Converts floats to decimals which allows for fixed width json."""
    if prec and isinstance(data, dict):
        return {k: float_to_decimal(v, prec) for k, v in data.items()}
    if prec and isinstance(data, float):
        return decimal.Decimal(("{:." + str(prec) + "f}").format(data))
    else:
        return data


def dump_json_stats(stats):
    """Converts stats dict into json string for logging."""
    stats = float_to_decimal(stats)
    stats_json = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    return stats_json


def time_string(seconds):
    """Converts time in seconds to a fixed-width string format."""
    days, rem = divmod(int(seconds), 24 * 3600)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


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
    """Measures train stats."""

    def __init__(self, ep_iters, num_ep, log_period=1):
        self.ep_iters = ep_iters
        self.num_ep = num_ep
        self.log_period = log_period
        self.max_iter = num_ep * ep_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(log_period)
        self.loss_total = 0.0
        self.mse = ScalarMeter(log_period)
        self.mse_total = 0.0
        self.vb = ScalarMeter(log_period)
        self.vb_total = 0.0
        self.lr = None

    def reset(self, timer=False):
        self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.mse.reset()
        self.mse_total = 0.0
        self.vb.reset()
        self.vb_total = 0.0
        self.lr = None

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def record_stats(self, loss_dict, lr):
        for k, v in loss_dict.items():
            meter = f'{k}_total'
            setattr(self, meter, getattr(self, meter) + v)
            getattr(self, k).add_value(v)
        self.lr = lr

    def get_iter_stats(self, cur_epoch, cur_iter):
        cur_iter_total = cur_epoch * self.ep_iters + cur_iter
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        return {
            "_type":  "train_iter",
            "epoch": "{}/{}".format(cur_epoch, self.num_ep),
            "iter": "{}/{}".format(cur_iter, self.ep_iters),
            "time_avg": self.iter_timer.average_time,
            "eta": time_string(eta_sec),
            "loss": self.loss.get_win_median(),
            "mse": self.mse.get_win_median(),
            "vb": self.vb.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage))
        }

    def log_iter_stats(self, cur_epoch, cur_iter):
        if cur_iter % self.log_period == 0 and cur_iter != 0:
            stats = self.get_iter_stats(cur_epoch, cur_iter)
            #print(dump_json_stats(stats))
            wandb.log(
                data={
                    'iter/train_loss': stats["loss"],
                    'iter/train_mse': stats["mse"],
                    #'iter/train_vb': stats["vb"],
                    'iter/lr': stats["lr"],
                    'epoch': cur_epoch + (cur_iter / self.ep_iters)

                }
            )

    def get_epoch_stats(self, cur_epoch):
        cur_iter_total = cur_epoch * self.ep_iters
        eta_sec = self.iter_timer.average_time * (self.max_iter - cur_iter_total)
        mem_usage = gpu_mem_usage()
        avg_loss = self.loss_total / self.ep_iters
        avg_mse = self.mse_total / self.ep_iters
        avg_vb = self.vb_total / self.ep_iters
        return {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch, self.num_ep),
            "time_epoch": self.iter_timer.average_time * self.ep_iters,
            "eta": time_string(eta_sec),
            "loss": avg_loss,
            "mse": avg_mse,
            "vb": avg_vb,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage))
        }

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        print(dump_json_stats(stats))
        wandb.log(
            data={
                'epoch/train_loss': stats["loss"],
                'epoch/train_mse': stats["mse"],
                #'epoch/train_vb': stats["vb"],
                'epoch/lr': stats["lr"],
                'epoch': cur_epoch
            }
        )


class TestMeter(object):
    """Measures test stats."""

    def __init__(self, ep_iters, num_ep, log_period=10):
        self.ep_iters = ep_iters
        self.num_ep = num_ep
        self.log_period = log_period
        self.loss_total = 0.0
        self.mse_total = 0.0
        self.vb_total = 0.0

    def reset(self, timer=False):
        self.loss_total = 0.0
        self.mse_total = 0.0
        self.vb_total = 0.0

    def record_stats(self, loss_dict):
        self.loss_total += loss_dict['loss']
        if 'mse' in loss_dict:
            self.mse_total += loss_dict['mse']
        if 'vb' in loss_dict:
            self.vb_total += loss_dict['vb']

    def get_epoch_stats(self, cur_epoch):
        avg_loss = self.loss_total / self.ep_iters
        avg_mse = self.mse_total / self.ep_iters
        avg_vb = self.vb_total / self.ep_iters
        return {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch, self.num_ep),
            "loss": avg_loss,
            "mse": avg_mse,
            "vb": avg_vb
        }

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        print(dump_json_stats(stats))
        wandb.log(
            data={
                'epoch/test_loss': stats["loss"],
                'epoch/test_mse': stats["mse"],
                #'epoch/test_vb': stats["vb"]
                'epoch': cur_epoch
            }
        )
