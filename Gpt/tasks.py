"""
This file contains metadata for different supervised learning and reinforcement learning tasks.

To define a new task, you should add a new sub-dictionary to TASK_METADATA below with the following keys:

(1) task_test_fn, a Python function of the form f(x, y, z, ..., nn.Module) that outputs loss/reward/etc.

(2) constructor, a Python function that returns a (random-init) nn.Module with the architecture G.pt is learning

(3) data_fn, a Python function that returns (x, y, z, ...); i.e., anything you want to cache for task_test_fn
             Note that data_fn must return a tuple or list, even if it returns only one thing

(4) minimize, a boolean indicating if the goal is to minimize the output of task_test_fn (False = maximize)

(5) best_prompt, the "best" loss/error/return/etc. you want to prompt G.pt with for one-step training

(6) recursive_prompt, the loss/error/return/etc. you want to prompt G.pt with for recursive optimization

(Optional) You can also include an 'aug_fn' key that maps to a function that performs a loss-preserving augmentation
           on the neural network parameters directly.

Whatever key you choose for your new task should be passed in with dataset.name.

See below for examples.
"""

import data_gen.train_mnist
import data_gen.train_cifar10

try:
    import data_gen.train_rl
except RuntimeError:
    print("WARNING: data_gen.train_rl not imported")


TASK_METADATA = {
    "mnist_loss": {
        "task_test_fn": data_gen.train_mnist.test_epoch,
        "constructor": lambda: data_gen.train_mnist.MLP(w_h=10),
        "data_fn": data_gen.train_mnist.unload_test_set,
        "aug_fn": data_gen.train_mnist.random_permute_mlp,
        "minimize": True,
        "best_prompt": 0.0,
        "recursive_prompt": 0.0
    },
    "mnist_error": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_mnist.test_epoch(*args, **kwargs)[1],
        "constructor": lambda: data_gen.train_mnist.MLP(w_h=10),
        "data_fn": data_gen.train_mnist.unload_test_set,
        "aug_fn": data_gen.train_mnist.random_permute_mlp,
        "minimize": True,
        "best_prompt": 5.0,
        "recursive_prompt": 5.0
    },
    "cifar10_loss": {
        "task_test_fn": data_gen.train_cifar10.test_epoch,
        "constructor": data_gen.train_cifar10.ConvNet,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 1.2,
        "recursive_prompt": 1.4
    },
    "cifar10_error": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        "constructor": data_gen.train_cifar10.ConvNet,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    },
    "cartpole": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_rl.vis_test_policy(*args, maxlen=200, **kwargs),
        "constructor": lambda: data_gen.train_rl.VisActorCritic("Cartpole", 4, 1),
        "data_fn": lambda: data_gen.train_rl.vis_create_env("Cartpole"),
        "aug_fn": data_gen.train_rl.random_permute_actor,
        "minimize": False,
        "best_prompt": 500.0,
        "recursive_prompt": 500.0
    },
}


def get(dataset_name, key):
    return TASK_METADATA[dataset_name][key]
