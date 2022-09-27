"""
The main Dataset class for training/testing on checkpoint data (parameters, metrics, etc.).
"""
import json
import os

from dataclasses import dataclass
from glob import glob
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from .database import Database
from .augment import random_permute_flat
from .normalization import get_normalizer
from Gpt.tasks import TASK_METADATA
from Gpt.download import find_checkpoints


@dataclass
class ParameterDataset(Dataset):
    dataset_dir: str = "checkpoint_datasets/mnist"  # Path to the checkpoint dataset
    dataset_name: str = "mnist_loss"                # Name of the dataset in tasks.py
    split: str = "train"                            # Split of the checkpoint dataset to use ("train" or "test")
    max_train_runs: int = 1000000                   # Maximum number of runs to train on (default: all)
    num_test_runs: int = 500                        # Number of runs in the test split
    target_epoch_size: int = 806400                 # Amount of data to train on per-"epoch" (806400 is arbitrary)
    train_metric: str = "avg_test_loss"             # Conditional metric for G.pt
    min_step_spacing: int = 1                       # Minimum spacing between starting and future checkpoints
    max_step_spacing: int = None                    # Maximum spacing between starting and future checkpoints
    normalizer_name: str = "openai"                 # Parameter normalization algorithm
    openai_coeff: float = 4.185                      # Scaling coefficient for the "openai" DALL-E 2 normalizer
    single_run_debug: bool = False                  # Option to debug with a single run
    min_val: float = None                            # Minimum value of parameters (usually passed to test dataset)
    max_val: float = None                            # Maximum value of parameters (usually passed to test dataset)
    permute_augment: bool = False                   # if True, applies permutation augmentation to parameters
    verify_done: bool = False                       # if True, filters runs that don't have a DONE.txt file
    download: bool = True                           # If True, auto-downloads and caches the checkpoint dataset

    def __post_init__(self):

        # Auto-download the dataset:
        if self.download:
            find_checkpoints(self.dataset_dir)

        # Find all the LDMB directories:
        lmdb_paths = sorted(list(glob(f'{self.dataset_dir}/*')))
        if len(lmdb_paths) == 0:
            raise FileNotFoundError

        # Filter out runs missing DONE.txt files if needed:
        if self.verify_done:
            lmdb_paths = [p for p in lmdb_paths if os.path.exists(os.path.join(p, "DONE.txt"))]

        # Build run to lmdb path map:
        run_lmdb_path = dict()
        for lmdb_path in lmdb_paths:
            lmdb_runs = os.path.basename(lmdb_path).split("_")
            for run in lmdb_runs:
                run_lmdb_path[run] = lmdb_path
        self.runs = list(run_lmdb_path.keys())

        # Perform the train/test split:
        assert self.split in ["train", "test"]
        assert self.num_test_runs < len(self.runs)
        if self.split == "train":
            self.runs = self.runs[:-self.num_test_runs] if self.num_test_runs > 0 else self.runs
            self.runs = self.runs[:self.max_train_runs]
        else:
            self.runs = self.runs[-self.num_test_runs:] if self.num_test_runs > 0 else []
        # Delete the runs we aren't using:
        self.run_lmdb_path = {run: run_lmdb_path[run] for run in self.runs}
        lmdb_paths = list(self.run_lmdb_path.values())
        print(f"(split={self.split}) number of runs: {len(self.runs)}")

        # Read run jsons containing various metadata:
        self.run_jsons = []
        for run in self.runs:
            lmdb_path = self.run_lmdb_path[run]
            json_path = os.path.join(lmdb_path, "runs.json")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            # Extract run json in case of fused jsons
            if run in json_data:
                json_data = json_data[run]
            self.run_jsons.append(json_data)
        assert len(self.runs) == len(self.run_jsons)

        # Build map of LMDB paths to LMDB objects:
        self.databases = {
            lmdb_path: Database(path=lmdb_path, map_size=10485760) for lmdb_path in lmdb_paths
        }
        self.num_runs = len(self.runs)
        self.lmdb_paths = lmdb_paths
        self.architecture = TASK_METADATA[self.dataset_name]['constructor']()
        self.parameter_sizes = self.get_database(0)[\
            f'{list(self.run_jsons[0]["checkpoints"].keys())[0]}_arch'].long().tolist()
        self.parameter_names = list(self.architecture.state_dict().keys())
        assert len(self.parameter_names) == len(self.parameter_sizes)
        self.num_checkpoints = [len(run_json['checkpoints']) for run_json in self.run_jsons]
        self.generator = None

        # Setup parameter normalization:
        reduce_fn = min if TASK_METADATA[self.dataset_name]['minimize'] else max
        self.optimal_test_loss = self.reduce_metadata(
            f'optimal_{self.train_metric}'.replace('_avg', ''), reduce_fn=reduce_fn
        )
        self.min_val, self.max_val = self.get_range(normalize=False)
        self.normalizer = get_normalizer(self.normalizer_name, openai_coeff=self.openai_coeff,
                                         min_val=self.min_val, max_val=self.max_val, dataset=self)
        print(f"(split={self.split}) using normalizer={self.normalizer.message()}")
        print(f'(split={self.split}) max-val: {self.max_val}, min-val: {self.min_val}')

        # Setup parameter augmentation if needed:
        self.make_aug_fn()

        # Ensure that epoch_size is perfectly divisible by the number of runs:
        assert self.target_epoch_size >= self.num_runs, \
            f"target_epoch_size ({self.target_epoch_size}) " \
            f"is less than the number of runs ({self.num_runs})"
        self.epoch_size = self.num_runs * (self.target_epoch_size // self.num_runs)

    def get_database(self, run_index):
        return self.databases[self.run_lmdb_path[self.runs[run_index]]]

    def make_aug_fn(self):
        task_dict = TASK_METADATA[self.dataset_name]
        self.use_augment = self.permute_augment and 'aug_fn' in task_dict
        if self.use_augment:
            self.task_aug_fn = task_dict['aug_fn']
            print(f'(split={self.split}) Using augmentation')
        else:
            print(f'(split={self.split}) NOT using augmentation')

    def aug_fn(self, p, seed=None):
        if self.use_augment:
            return random_permute_flat(p, self.architecture, seed, self.task_aug_fn)
        else:
            return p

    def get_range(self, normalize=True):
        if self.min_val is None and self.max_val is None:
            min_val = self.reduce_metadata('min_parameter_val', reduce_fn=min)
            max_val = self.reduce_metadata('max_parameter_val', reduce_fn=max)
        else:
            min_val, max_val = self.min_val, self.max_val
        if normalize:
            # If normalize=True, this returns the range of normalized parameter values
            assert hasattr(self, "normalizer"), "normalizer hasn't been instantiated yet"
            min_val, max_val = self.normalizer.get_range(min_val, max_val)
        return min_val, max_val

    def normalize(self, weights):
        return self.normalizer.normalize(weights)

    def unnormalize(self, normalized_weights):
        return self.normalizer.unnormalize(normalized_weights)

    def reduce_metadata(self, key, reduce_fn=max):
        # Applies a reduction function over all runs in this split
        return reduce_fn(run_json['metadata'][key] for run_json in self.run_jsons)

    def get_run_losses(self, run_index: int):
        if self.single_run_debug:
            run_index = 0
        metadata = self.run_jsons[run_index]['metadata']
        test_losses = torch.tensor(metadata[self.train_metric])
        return test_losses

    def get_run_network(self, run_index, iter=0, normalize=True, augment=False):
        if self.single_run_debug:
            run_index = 0
        run_checkpoint_names = list(self.run_jsons[run_index]['checkpoints'].keys())
        checkpoint_name = run_checkpoint_names[iter]
        database = self.get_database(run_index)
        parameters = database[checkpoint_name]
        if normalize:
            parameters = self.normalize(parameters)
        if augment:
            parameters = self.aug_fn(parameters)
        return parameters

    def __getitem__(self, run_index: int) -> Dict[str, Any]:
        run_index = run_index % len(self.runs)
        if self.single_run_debug:
            run_index = 0

        run_name = self.runs[run_index]
        run_checkpoints = self.run_jsons[run_index]['checkpoints']

        num_checkpoints = self.num_checkpoints[run_index]
        max_step_spacing = (num_checkpoints - 1) if self.max_step_spacing is None else self.max_step_spacing
        checkpoint_names = list(self.run_jsons[run_index]['checkpoints'].keys())
        step_spacing = torch.randint(
            low=self.min_step_spacing,
            high=max_step_spacing + 1,
            size=(),
            generator=self.generator,
        ).item()

        checkpoint_index_0 = torch.randint(
            low=0,
            high=num_checkpoints - step_spacing,
            size=(),
            generator=self.generator,
        ).item()
        if self.single_run_debug:
            checkpoint_index_0 = 0
        checkpoint_index_1 = checkpoint_index_0 + step_spacing

        run_checkpoint_name_0 = checkpoint_names[checkpoint_index_0]
        run_checkpoint_name_1 = checkpoint_names[checkpoint_index_1]

        run_metrics_0 = run_checkpoints[run_checkpoint_name_0]
        run_metrics_1 = run_checkpoints[run_checkpoint_name_1]

        database = self.get_database(run_index)
        parameters_0 = database[run_checkpoint_name_0]
        parameters_1 = database[run_checkpoint_name_1]

        parameters_0 = self.normalize(parameters_0)
        parameters_1 = self.normalize(parameters_1)

        parameters_0, parameters_1 = self.aug_fn((parameters_0, parameters_1), seed=None)

        outputs = {
            "parameters_0": parameters_0,
            "parameters_1": parameters_1,
            "checkpoint_key_0": run_checkpoint_name_0,
            "checkpoint_key_1": run_checkpoint_name_1,
            "run_name": run_name,
            "step_spacing": step_spacing,
        }

        for metric in [self.train_metric]:
            outputs[f"{metric}_0"] = run_metrics_0[metric]
            outputs[f"{metric}_1"] = run_metrics_1[metric]

        return outputs

    def __len__(self) -> int:
        return self.epoch_size
