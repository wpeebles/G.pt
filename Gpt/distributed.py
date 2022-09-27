#!/usr/bin/env python3

"""Distributed helpers (source: https://github.com/facebookresearch/pycls/blob/main/pycls/core/distributed.py)."""

import os
import random

import submitit
import torch


# Make work w recent PyTorch versions (https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_THREADING_LAYER"] = "GNU"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, fun, cfg_state):
        self.cfg_state = cfg_state
        self.port = port
        self.fun = fun

    def __call__(self):
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        setup_distributed(self.cfg_state)
        self.fun()


def is_main_proc(local=False):
    """
    Determines if the current process is the main process.

    Main process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the main role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered main.

    If local==True, then check if the current process is the main on the current node.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


def scaled_all_reduce(cfg, tensors):
    """
    Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    if cfg.num_gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.num_gpus)
    return tensors


def setup_distributed(cfg_state):
    """
    Initialize torch.distributed and set the CUDA device.

    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.

    This is run inside a new process, so the cfg is reset and must be set explicitly.
    """
    cfg.defrost()
    cfg.update(**cfg_state)
    cfg.freeze()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend=cfg.DIST_BACKEND)
    torch.cuda.set_device(local_rank)


def single_proc_run(local_rank, fun, main_port, cfg_state, world_size):
    """Executes fun() on a single GPU in a multi-GPU setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    setup_distributed(cfg_state)
    fun()


def multi_proc_run(num_proc, fun):
    """Run a single or multi GPU job locally on the current node."""
    launch = cfg.LAUNCH
    if launch.MODE in ["submitit_local", "slurm"]:
        # Launch fun() using submitit either locally or on SLURM
        use_slurm = launch.MODE == "slurm"
        executor = submitit.AutoExecutor if use_slurm else submitit.LocalExecutor
        kwargs = {"slurm_max_num_timeout": launch.MAX_RETRY} if use_slurm else {}
        executor = executor(folder=cfg.OUT_DIR, **kwargs)
        num_gpus_per_node = min(cfg.NUM_GPUS, cfg.MAX_GPUS_PER_NODE)
        executor.update_parameters(
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_gpus_per_node,
            cpus_per_task=launch.CPUS_PER_GPU,
            nodes=max(1, cfg.NUM_GPUS // cfg.MAX_GPUS_PER_NODE),
            timeout_min=launch.TIME_LIMIT,
            name=launch.NAME,
            slurm_qos=launch.QOS,
            slurm_comment=launch.COMMENT,
            slurm_additional_parameters={"mail-user": launch.EMAIL, "mail-type": "END"},
        )
        main_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        job = executor.submit(SubmititRunner(main_port, fun, cfg))
        print("Submitted job_id {} with out_dir: {}".format(job.job_id, cfg.OUT_DIR))
        if not use_slurm:
            job.wait()
    elif num_proc > 1:
        main_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        mp_runner = torch.multiprocessing.start_processes
        args = (fun, main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="fork")
    else:
        fun()


def get_rank():
    if not torch.distributed.is_available():
        return 0

    if not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


def synchronize():

    if get_world_size() == 1:
        return

    torch.distributed.barrier()


def get_world_size():
    if not torch.distributed.is_available():
        return 1

    if not torch.distributed.is_initialized():
        return 1

    return torch.distributed.get_world_size()


def all_gather(input, cat=True, dim=0):
    if get_world_size() == 1:
        if cat:
            return input
        else:
            return input.unsqueeze(dim)
    input_list = [torch.zeros_like(input) for _ in range(get_world_size())]
    synchronize()
    torch.distributed.all_gather(input_list, input, async_op=False)
    if cat:
        inputs = torch.cat(input_list, dim=dim)
    else:
        inputs = torch.stack(input_list, dim=dim)
    return inputs


def rank0_to_all(input, cat=True):
    input = all_gather(input, cat)
    rank0_input = input[0]
    return rank0_input
