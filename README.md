## Learning to Learn with Generative Models of Neural Network Checkpoints<br><sub>Official PyTorch Implementation</sub>

### [Paper](http://arxiv.org/abs/2209.12892) | [Project Page](https://www.wpeebles.com/Gpt)

![Overview of G.pt](images/Gpt_lightmode.gif#gh-light-mode-only)
![Overview of G.pt](images/Gpt_darkmode.gif#gh-dark-mode-only)

This repo contains training, evaluation, and visualization code for our recent paper exploring 
loss-conditional diffusion models of neural network parameters. 

> [**Learning to Learn with Generative Models of Neural Network Checkpoints**](https://www.wpeebles.com/Gpt)<br>
> [William Peebles*](https://www.wpeebles.com), [Ilija Radosavovic*](https://people.eecs.berkeley.edu/~ilija/),
> [Tim Brooks](https://www.timothybrooks.com), [Alexei A. Efros](http://people.eecs.berkeley.edu/~efros/), 
> [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)
> <br>University of California, Berkeley<br>

Our generative models are conditioned on a starting parameter vector, a starting loss/error/return and a 
_prompted_ loss/error/return. With these inputs, we can sample an updated parameter vector that ideally achieves the 
prompt. We call our model G.pt (G and .pt refer to generative models and checkpoint extensions, respectively). The core of 
G.pt is a transformer model that operates over sequences of parameters from the input neural network
parameters. Similar to ViTs, G.pt leverages very few domain-specific inductive biases (only in tokenization and data augmentation). 
The transformer is trained as a diffusion model directly in parameter space. After training, G.pt
can optimize neural networks from random initialization in one step by prompting for a small loss/error or high return. 
In this paper, we introduce G.pt models for optimizing MNIST MLPs, CIFAR-10 CNNs and Cartpole Gaussian MLPs. 

This repository contains:

* ⚡️ Five pre-trained G.pt DDPM Transformers for vision and RL tasks
* 🪐 A dataset containing over 23M neural net checkpoints across 100K+ training runs
* 💥 Training and testing scripts for G.pt models

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/wpeebles/G.pt.git
cd G.pt
pip install -e .
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate G.pt
```

If you opt to use your own environment, you'll need Python 3.8 in order to run the IsaacGym RL simulator (newer versions
of Python may have compatibility issues). You'll also want to set up a [Weights & Biases](https://wandb.ai/site) account 
since some of our visualization code uses it.

Finally, in order to train or evaluate RL G.pt models, you'll need to install IsaacGym. 
[Download it here](https://developer.nvidia.com/isaac-gym), then install it:

```bash
cd /path/to/isaac-gym/python
pip install -e .
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/anaconda3/envs/G.pt/lib
```

## Checkpoint Pre-training Data

We provide three checkpoint datasets, in aggregate containing over 23M checkpoints from 100K+ training runs. Each 
individual checkpoint contains neural network parameters and any useful task-specific metadata (e.g., test losses and 
errors for classification, episode returns for RL). If you run our G.pt testing scripts (explained 
[below](#evaluating-gpt-models)), the relevant checkpoint data will be auto-downloaded. Or, you can 
download all three checkpoint datasets (and the five pre-trained G.pt models) by running:

```python
python Gpt/download.py
```

This will store all three datasets in a folder named `checkpoint_datasets`. The breakdown for each dataset is 
as follows:

| Checkpoint Dataset     | # Checkpoints | # Runs  | # Checkpoints/Run | Storage (GB) |
|------------------------|---------------|---------|-------------------|--------------|
| MNIST MLPs             | 2.1M          | 10728   | 200               | 68           |
| CIFAR-10 CNNs          | 11.3M         | 56840   | 200               | 275          |
| Cartpole Gaussian MLPs | 10M           | 50026   | 200               | 82           | 

Additional information about the datasets can be found in our [paper](http://arxiv.org/abs/2209.12892).

## Evaluating G.pt Models

![One step training](images/cartpole_lightmode.gif#gh-light-mode-only)
![One step training](images/cartpole_darkmode.gif#gh-dark-mode-only)

**Using our pre-trained G.pt models.** We provide five pre-trained G.pt models. They can be used by setting the config's 
`resume_path` to one of `cartpole.pt`, `mnist_loss.pt`, `mnist_error.pt`, `cifar_loss.pt` or `cifar_error.pt`. If you use 
one of these values, the relevant model will be automatically downloaded and cached in the `pretrained_models` folder.

You can evaluate G.pt models by running [`main.py`](main.py) with the `test_only=True` flag. We provide several
testing configs [here](configs/test). For example, to create one-step/recursive optimization curves and compute prompt
alignment scores for our loss-conditional MNIST model, you can run:

```python
python main.py --config-path configs/test --config-name mnist_loss.yaml num_gpus=N
```

where `N` is the number of GPUs to distribute the evaluation across. Running this command will generate plots in 
Weights & Biases. There are several options in the configs you can change.

We also include a [`playground.py`](playground.py) script which gives a more minimal example of loading and running 
G.pt models on one GPU. It can be used to generate latent walks through parameter space for our CIFAR-10 G.pt models:

```python
python playground.py  --config-name cifar_error.yaml
```

![Latent walk](images/latent_walk_lightmode.gif#gh-light-mode-only)
![Latent walk](images/latent_walk_darkmode.gif#gh-dark-mode-only)

## Training G.pt Models

The main entrypoint for training new G.pt models is [`main.py`](main.py). You can find our training configs
[here](configs/train). Example usage to train a G.pt model using `N` GPUs:

```python
python main.py --config-name mnist_loss.yaml num_gpus=N
```

## Creating New Tasks

To add a new task, you need to update the `TASK_METADATA` dictionary in [`tasks.py`](Gpt/tasks.py) with a new entry. 
You'll need to come up with a name for the task which will be the new key. The corresponding value should be a 
dictionary with the following items: 

(1) `task_test_fn`, 
a function mapping any needed inputs and an `nn.Module` instance to a loss/error/return/etc. Your function should 
explicitly take as input anything that is expensive to construct multiple times (e.g., datasets, dataloaders, simulators, etc.). 
You will specify any of these expensive inputs via `data_fn` below. Make sure the `nn.Module` is the last input to this function.

(2) `constructor`, a function 
that constructs a randomly-initialized `nn.Module` with the correct architecture. This constructor needs to produce the correct architecture when called without any arguments.

(3) `data_fn`, a function that outputs 
any inputs needed to call `task_test_fn` (besides the input `nn.Module`) which should be cached. For example, data and 
environment instances should be returned by `data_fn`, and they will then be automatically passed to `task_test_fn` whenever
it is invoked. This avoids expensive re-instantiation of these objects everytime we want to call `task_test_fn` (which is usually a lot of times). 
`data_fn` must always return a tuple or list, even if it returns only one thing or nothing.

(4) `minimize`, a boolean that indicates if the goal is to minimize or maximize the output of `task_test_fn`.

(5) `best_prompt`, a float representing the "best" loss/error/return/etc. you want to prompt G.pt with for 
one-step optimization.

(6) `recursive_prompt`, a float representing the loss/error/return/etc. you want to repeatedly prompt G.pt with when 
performing recursive optimization.

(Optional, but recommended) You can also include an `aug_fn` key that maps to a function that performs a loss-preserving 
augmentation on the neural network parameters directly.

Finally, make sure you pass the name of your new task via `dataset.name`.

## Creating New Checkpoint Datasets

We provide several scripts to facilitate checkpoint generation in the [`data_gen`](data_gen) folder. We include example 
files that generate supervised learning and reinforcement learning checkpoints. You can use the
[`train_batch.py`](data_gen/train_batch.py) script to indefinitely launch single-GPU task-level training jobs on a node 
to collect a large number of checkpoints. After saving enough checkpoints, you'll want to filter out any with bad parameter 
values (e.g., NaNs). You can do this with [prepare_checkpoints.py](Gpt/data/prepare_checkpoints.py) (be sure to update 
it with a path to your new directory of checkpoints):

```python
python Gpt/data/prepare_checkpoints.py
```

`prepare_checkpoints.py` will also compute the variance over parameter values in your dataset, which you can pass via 
`dataset.openai_coeff` in your training/testing config file in order to normalize the parameters before being 
processed by G.pt.

## BibTeX

```bibtex
@article{Peebles2022,
  title={Learning to Learn with Generative Models of Neural Network Checkpoints},
  author={William Peebles and Ilija Radosavovic and Tim Brooks and Alexei Efros and Jitendra Malik},
  year={2022},
  journal={arXiv preprint arXiv:2209.12892},
}
```

## Acknowledgments
We thank Anastasios Angelopoulos, Shubham Goel, Allan Jabri, Michael Janner, Assaf Shocher, Aravind Srinivas, 
Matthew Tancik, Tete Xiao, Saining Xie and Jun-Yan Zhu for helpful discussions. William Peebles and Tim Brooks are 
supported by the NSF Graduate Research Fellowship. Additional support provided by the DARPA 
program on Machine Common Sense.

This codebase borrows from OpenAI's diffusion repos, most notably [ADM](https://github.com/openai/guided-diffusion), and 
Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT). We thank the authors for open-sourcing their work.
