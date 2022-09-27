import argparse
import atexit
import subprocess
import time
import os
import copy
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch training")
    parser.add_argument("--runs-per-gpu", type=int, default=1)
    parser.add_argument("--cmd", type=str, default='train_mnist.py')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')
    processes = [[None for _ in range(args.runs_per_gpu)] for _ in range(num_gpus)]
    env = os.environ.copy()
    envs = []
    for gpu_ix in range(num_gpus):
        env_copy = copy.deepcopy(env)
        env_copy["CUDA_VISIBLE_DEVICES"] = str(gpu_ix)
        envs.append(env_copy)

    def terminate_processes():
        for gpu in range(num_gpus):
            for process in processes[gpu]:
                if process is not None:
                    process.terminate()

    atexit.register(terminate_processes)

    while True:

        for gpu in range(num_gpus):
            for run in range(args.runs_per_gpu):
                process = processes[gpu][run]

                if process is None or process.poll() is not None:
                    process = subprocess.Popen(["python", args.cmd], env=envs[gpu])
                    processes[gpu][run] = process
            time.sleep(5)
        time.sleep(60)
