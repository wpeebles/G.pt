"""
Contains classes and function for generating G.pt visuals and evaluation metrics.
"""
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os

from Gpt.tasks import TASK_METADATA
from Gpt.diffusion.gaussian_diffusion import GaussianDiffusion
from Gpt.distributed import all_gather, synchronize, is_main_proc, rank0_to_all

from typing import Callable, Any, Dict


class VisMonitor:

    """
    This class is used to create Weights & Biases plots that show G.pt performance, as measured by the
    success of its generated networks. There are two types of graphs VisMonitor can generate:
        (1) DVO Curves (DVO = Desired Versus Observed), like Figure 5 in our paper.
        (2) Recursive Prompting Curves, like Figure 6 in our paper.
    In addition to (1), VisMonitor also computes Prompt Alignment scores for each DVO plot.
    """

    def __init__(
            self,
            dataset_name,            # string name of dataset
            task_net_dict,           # dictionary mapping a string to a (N, D) tensor of network that G.pt will process
            unnormalize_fn,          # function that "unnormalizes" sampled neural network parameters
            net_mb_size=1,           # minibatch size at which input networks are processed
            vis_recursion=True,      # if True, generates additional plots that visualize recursive prompting
            vis_period=2500,         # Number of training "epochs" between visualizations
            delay_test_fn=True,      # This argument is only used to avoid issues with IsaacGym
            dvo_steps=20,            # Number of prompts to use for creating DVO curves
            prompt_start_coeff=1.0,  # Specifies the "worst" loss/error/return G.pt will be prompted with for DVO
            thresholding="none",     # "none" or "static"; thresholding alg. to use after each diffusion sampling step
            param_range=None,        # A tuple of the (min, max) values that unnormalized network parameters can have
    ):
        self.metadata = TASK_METADATA[dataset_name]
        self.inp_data = self.metadata['data_fn']()  # Instantiates data (MNIST, CIFAR-10) or RL environments (IsaacGym)
        self.unnormalize_fn = unnormalize_fn
        if not delay_test_fn:
            self.create_test_fn()
        else:  # This is a hack to avoid issues with IsaacGym; we will manually call create_test_fn() later
            self.test_function = None
        self.create_synth_fn(thresholding, param_range)
        self.vis_period = vis_period
        self.task_net_dict = task_net_dict
        self.early_epochs_to_monitor = {1}  # Hard-coded epochs after which visuals are generated
        self.vis_recursion = vis_recursion
        self.net_mb_size = net_mb_size
        self.dataset_name = dataset_name
        self.dvo_steps = dvo_steps
        self.prompt_start_coeff = prompt_start_coeff

    def create_synth_fn(self, thresholding="none", param_range=None):
        """
        Creates a function to sample new parameters from G.pt. This is mainly for convenience (so we don't have to
        pass in the thresholding and param_range arguments later on below).
        """
        self.synth_fn = lambda *args: synth(*args, param_range=param_range, thresholding=thresholding)

    def create_test_fn(self):
        """
        Creates a function that computes the loss/error/return of a neural network on a particular dataset.
        The returned function takes a (1, D) tensor of neural network parameters as input and outputs a scalar.
        """
        assert self.unnormalize_fn is not None

        def test_function(pred):  # pred is a (1, D) unnormalized neural network parameter tensor
            modulified = moduleify(pred, self.metadata['constructor'], self.unnormalize_fn).to('cuda')
            val = self.metadata['task_test_fn'](*self.inp_data, modulified)
            if isinstance(val, (tuple, list)):
                val = val[0]
            return val

        self.test_function = test_function

    def vis_model(self, diffusion, model, epoch, trn_set_losses_dict, gt_trajectories, optimal_loss, exp_name=None):
        """
        Generate visuals for the input G.pt model.
        """
        assert self.test_function is not None
        if epoch % self.vis_period == 0 or epoch in self.early_epochs_to_monitor:
            model.eval()
            test_metric = visualize_G(
                self.synth_fn, diffusion, model, epoch, self.task_net_dict, self.test_function,
                self.metadata, self.net_mb_size, optimal_loss, trn_set_losses_dict, gt_trajectories,
                self.vis_recursion, exp_name,
                dvo_steps=self.dvo_steps, prompt_start_coeff=self.prompt_start_coeff
            )
        else:
            test_metric = None
        return test_metric


@torch.inference_mode()
def visualize_G(
        synth_fn: Callable,             # see create_synth_fn() above
        diffusion: GaussianDiffusion,   # The diffusion process to use for sampling
        G: nn.Module,                   # The G.pt model to visualize
        epoch: int,                     # The current training epoch
        task_net_dict: Dict[str, torch.Tensor],  # dictionary mapping a string to a (*, D) tensor of input nets for G.pt
        test_fn: Callable,          # the function mapping a raw G.pt sample to a downstream task loss/error/return/etc.
        metadata: Dict[str, Any],   # contains various task-specific quantities needed for visualization
        net_mb_size: int = 1,       # the number of networks to process in one minibatch
        optimal_loss_dict: Dict[str, float] = None,             # the best possible loss/error/return
        trn_set_losses_dict: Dict[str, torch.Tensor] = None,   # if specified, uses these values to prompt G.pt
        gt_trajectories: torch.Tensor = None,                  # compare against SGD/Adam trajectories
        vis_recursion: bool = True,         # if True, generates additional plots that visualize recursive prompting
        exp_name: str = None,               # if specified, saves plot data locally to disk based on exp_name
        **evolve_kwargs
) -> float:
    """
    This function creates several visualizations of the input G.pt neural network.
    DVO Curves, Recursive Prompting Curves and Prompt Alignment scores are logged to Weights & Biases.
    Returns the Prompt Alignment score on the checkpoint test split (useful for saving
    the best checkpoint during training).
    """
    print("Visualizing G.pt...")
    # zero = repeatedly ask G.pt to generate 0 loss network
    # step = ask for slightly smaller loss instead
    interp_algs = ['zero'] if vis_recursion else []
    metric = None  # The metric to return (e.g., the prompt alignment score)

    # Usually, we have two items in the task_net_dict dictionary: a "training" batch of networks and a "test" batch.
    # We generate visuals for each group separately (hence the for-loop below).
    for task_key, task_nets in task_net_dict.items():  # This is iterating over groups of task_nets

        # (1) Optionally generate recursive prompting curves:
        _, observed = probe_G(
            recursive_prompting, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn, metadata,
            interp_algs=interp_algs, **evolve_kwargs)  # (num_interp_algs, num_task_nets, T)

        # (2) Generate DVO curves with regularly-sampled loss/error/return prompts:
        queried, response = aggregated_evolve(
            one_step_prompting, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn,
            metadata, **evolve_kwargs)  # (num_task_nets, T)

        # (3) Generate DVO curves with *manually-specified* loss/error/return prompts:
        if task_key in trn_set_losses_dict:
            trn_set_queried, trn_set_response = aggregated_evolve(
                one_step_prompting, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn, metadata,
                desired_losses=trn_set_losses_dict[task_key], **evolve_kwargs)  # (num_task_nets, T)

        # (4) Plot the data in Weights & Biases:
        if is_main_proc():

            # Used for generating visuals:
            optimal_loss = optimal_loss_dict[task_key] \
                if optimal_loss_dict is not None and task_key in optimal_loss_dict else None

            # (4a) Plot the DVO curves with regularly-sampled prompts (and record the prompt alignment score):
            prompt_alignment_score = plot_desired_versus_observed(
                queried, response, epoch, tag=task_key, title_postfix=task_key,
                optimal_loss=optimal_loss, filepath=exp_name
            )

            # (4b) Plot the DVO curves with manually-specified prompts:
            if task_key in trn_set_losses_dict:
                plot_desired_versus_observed(
                    trn_set_queried, trn_set_response, epoch, tag=task_key,
                    title_postfix=f'{task_key}, Loss Deltas from Training Set',
                    optimal_loss=optimal_loss, filepath=exp_name
                )

            # (4c) Optionally plot recursive prompting curves:
            for observed_i, interp_alg_i in zip(observed, interp_algs):
                plot_time_versus_observed(
                    observed_i, tag=task_key, title_postfix=interp_alg_i, optimal_loss=optimal_loss,
                    gt_trajectories=gt_trajectories, filepath=exp_name
                )

            # (4d) Save the prompt alignment score on test set checkpoints:
            if 'test' in task_key:
                metric = torch.tensor(prompt_alignment_score, dtype=torch.float, device='cuda').view(1)

    # Distribute the prompt alignment score to all processes in DDP:
    if is_main_proc() and metric is None:
        metric = torch.tensor(float("inf"), dtype=torch.float, device='cuda').view(1)
    elif not is_main_proc():
        # Create a dummy value that will be overwritten by the value from the main process:
        metric = torch.empty(1, dtype=torch.float, device='cuda').view(1)
    metric = rank0_to_all(metric, cat=False).item()
    return metric


def create_thresholding_fn(thresholding, param_range):
    """
    Creates a function that thresholds after each diffusion sampling step.

    thresholding = "none": No thresholding.
    thresholding = "static": Clamp the sample to param_range.
    """

    if thresholding == "none":
        def denoised_fn(x):
            return x
    elif thresholding == "static":
        def denoised_fn(x):
            return torch.clamp(x, param_range[0], param_range[1])
    else:
        raise NotImplementedError

    return denoised_fn


def synth(
    diffusion,
    G,
    loss_target,        # The prompted loss/error/return: shape (N, 1)
    loss_prev,          # The starting loss/error/return: shape (N, 1)
    w_prev,             # The starting parameter vector: shape (N, D)
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process.
    Specifically, this function draws a sample from p(theta^*|prompt_loss,starting_loss,starting_theta).
    """
    assert loss_target.size(0) == loss_prev.size(0) == w_prev.size(0)

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
        # print(f"Using thresholding={thresholding} with min_val={param_range[0]}, max_val={param_range[1]}")
    else:
        denoised_fn = None

    model_kwargs = {
        'loss_target': loss_target,
        'loss_prev': loss_prev,
        'x_prev': w_prev
    }

    shape = w_prev.shape
    sample = diffusion.p_sample_loop(
        G,
        shape,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device='cuda',
        denoised_fn=denoised_fn,
        **p_sample_loop_kwargs
    )

    return sample


def r_squared(preds, targets):
    """
    Computes R^2 for a batch of predictions and targets and then averages over the batch dimension.
    """
    assert preds.size() == targets.size()
    assert preds.dim() > 1
    avg = targets.mean(dim=1, keepdims=True)
    var = (targets - avg).pow(2).sum(dim=1)
    err = (targets - preds).pow(2).sum(dim=1)
    r2 = 1.0 - err / (var + 1e-8)
    r2 = r2.mean().item()  # take average of R2s over batch dimension
    return r2


@torch.inference_mode()
def probe_G(
    evolve_fn, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn, metadata,
    interp_algs, **evolve_kwargs
):
    """
    Runs one-step or recursive prompting on a batch of nets, trying out different interp_algs.
    Basically a wrapper for aggregated_evolve below.
    """
    if len(interp_algs) == 0:
        return [], []
    desired_loss_stats, observed_loss_stats = [], []
    for interp_alg in interp_algs:
        desired_random, observed_random = aggregated_evolve(
            evolve_fn, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn,
            metadata, interp_alg=interp_alg, **evolve_kwargs
        )
        desired_loss_stats.append(desired_random)
        observed_loss_stats.append(observed_random)
    desired_loss_stats = torch.stack(desired_loss_stats)  # (num_interp_algs, num_task_nets, T)
    observed_loss_stats = torch.stack(observed_loss_stats)  # (num_interp_algs, num_task_nets, T+1)
    return desired_loss_stats, observed_loss_stats


@torch.inference_mode()
def aggregated_evolve(
    evolve_fn, task_nets, net_mb_size, synth_fn, diffusion, G, test_fn, metadata, **evolve_kwargs
):
    """
    Runs one-step or recursive prompting on a batch of nets, distributed over several GPUs with DDP.
    Basically a wrapper for one_step_prompting and recursive_prompting below, where this handles batching and DDP.
    """
    desired_loss_stats, observed_loss_stats = [], []
    # Split input networks into batches if needed (uses less GPU memory):
    task_net_batches = torch.split(task_nets, net_mb_size, dim=0)
    tasknet_index = 0

    for task_net_batch in task_net_batches:
        desired, observed = evolve_fn(
            task_net_batch, synth_fn, diffusion, G, test_fn, metadata,
            tasknet_index=tasknet_index, **evolve_kwargs
        )
        desired_loss_stats.append(desired)
        observed_loss_stats.append(observed)
        tasknet_index += task_net_batch.size(0)

    desired_loss_stats = torch.cat(desired_loss_stats, 0)  # (num_task_nets, T)
    observed_loss_stats = torch.cat(observed_loss_stats, 0)  # (num_task_nets, T+1)

    # Aggregate results across GPUs:
    synchronize()
    desired_loss_stats = all_gather(desired_loss_stats, dim=0).cpu()
    observed_loss_stats = all_gather(observed_loss_stats, dim=0).cpu()
    return desired_loss_stats, observed_loss_stats


@torch.inference_mode()
def recursive_prompting(
    task_net, synth_fn, diffusion, G, test_fn, metadata, tasknet_index=0, device='cuda',
    interp_alg='zero', max_steps=2, step_divider=5, **ignore
):
    """
    Optimizes a batch of input neural networks with multiple updates sampled from G.pt recursively (akin to traditional
    iterative optimization).
    """
    num_nets = task_net.size(0)
    current_loss = batch_test_fn(test_fn, task_net)
    current_net = task_net
    desired_losses = []  # This will be length max_steps
    observed_losses = [current_loss]  # This will be length max_steps + 1
    pbar = range(max_steps)
    if is_main_proc():
        pbar = tqdm(pbar, leave=False, desc=f'recursive prompting (interp_alg={interp_alg})')
    sign = -1 if metadata['minimize'] else 1
    for step in pbar:
        if interp_alg == 'zero':
            desired_loss = scalar_expand(metadata['recursive_prompt'], num_nets)
        elif interp_alg == 'step':
            desired_loss = current_loss + sign * (metadata['best_prompt'] / float(step_divider))
        else:
            raise NotImplementedError
        desired_losses.append(desired_loss)
        current_net = synth_fn(diffusion, G, desired_loss, current_loss, current_net)
        current_loss = batch_test_fn(test_fn, current_net)
        observed_losses.append(current_loss)
    desired_losses = torch.stack(desired_losses, 1)  # (num_nets, max_steps)
    observed_losses = torch.stack(observed_losses, 1)  # (num_nets, 1 + max_steps)
    return desired_losses, observed_losses


@torch.inference_mode()
def one_step_prompting(
    task_net, synth_fn, diffusion, G, test_fn, metadata, tasknet_index=0, device='cuda', dvo_steps=20,
    desired_losses=None, prompt_start_coeff=1.0, **ignore
):
    """
    Optimizes a batch of input neural networks with one update sampled from G.pt.
    """
    num_nets = task_net.size(0)
    base_loss = batch_test_fn(test_fn, task_net)  # (N,)

    if desired_losses is None:
        # Regularly-sample prompts between the starting (current) loss/error/return and the best_prompt:
        start_loss = base_loss
        best_prompt = scalar_expand(metadata['best_prompt'], num_nets)  # (N,)
        desired_losses = batch_linspace(best_prompt, base_loss * prompt_start_coeff, steps=dvo_steps, device=device)  # (N, dvo_steps)
        if not metadata['minimize']:
            desired_losses = desired_losses.flip(1,)
    else:
        # Pass exact desired_losses you want to prompt G.pt with:
        desired_losses = desired_losses[tasknet_index: tasknet_index + num_nets]  # (N, steps)
        start_loss = desired_losses[:, 0].to(device)  # (N,)
        desired_losses = desired_losses[:, :dvo_steps].to(device)  # (N, dvo_steps)

    num_evos = desired_losses.size(1)
    desired_losses_in = desired_losses.flatten()  # (N * dvo_steps,)
    current_losses = start_loss.repeat_interleave(num_evos, dim=0)  # (N * dvo_steps)
    nets_in = task_net.repeat_interleave(num_evos, dim=0)  # (N * dvo_steps, 1)
    evolved_nets = synth_fn(
        diffusion, G, desired_losses_in, current_losses, nets_in
    )
    if is_main_proc():
        evolved_nets = tqdm(evolved_nets, leave=False, desc='one step prompting eval', total=evolved_nets.size(0))
    observed_losses = batch_test_fn(test_fn, evolved_nets, check_dims=not is_main_proc())
    observed_losses = observed_losses.view(num_nets, num_evos)
    observed_losses = torch.cat([base_loss.view(num_nets, 1), observed_losses], 1)  # (N, 1 + dvo_steps)
    return desired_losses, observed_losses


def batch_test_fn(test_fn, net_batch, return_as_tensor=True, check_dims=True, device='cuda'):
    """
    test_fn: A function that takes a (1, D) tensor representing a single neural net and returns a scalar
    net_batch: A (N, D) tensor representing a batch of neural nets
    return_as_tensor: If True, return a (N,) tensor of loss/errors/returns. If False, return a list of scalars.
    check_dims: If True, check that net_batch has the right shape. Otherwise, assume it does.
    device: The device to return the losses/errors/returns on (if return_as_tensor is True)

    Returns a (N,) tensor of loss/errors/returns if return_as_tensor is True, otherwise returns a list of scalars.
    """
    if check_dims:
        assert net_batch.dim() == 2
    result = [test_fn(net.unsqueeze(0)) for net in net_batch]
    if return_as_tensor:
        result = torch.tensor(result, dtype=torch.float, device=device)
    return result


def batch_linspace(start, end, **linspace_kwargs):
    """
    A version of torch.linspace that works on a batch of start/end inputs.
    """
    assert start.dim() == end.dim() == 1
    assert start.size(0) == end.size(0)
    result = [torch.linspace(s.item(), e.item(), **linspace_kwargs) for s, e, in zip(start, end)]
    result = torch.stack(result, 0)
    return result


def scalar_expand(scalar, num):
    """
    Expand a scalar to a (num,) tensor on GPU.
    """
    return torch.tensor(scalar, dtype=torch.float, device='cuda').view(1).repeat(num)


def lists2table(xs, ys, keys, xname, yname):
    """
    This function reformats data to be plotted on Weights & Biases.
    """
    assert len(xs) == len(ys) == len(keys)
    data = []
    # Official wandb code:
    for i, series in enumerate([list(zip(xs[i], ys[i])) for i in range(len(xs))]):
        for x, y in series:
            key = keys[i]
            data.append([x, key, y])
    table = wandb.Table(data=data, columns=[xname, "model", yname])
    return table


def plot_desired_versus_observed(
    desired_losses, observed_losses, epoch, tag='', title_postfix='', optimal_loss=None, filepath=None
):
    """
    Plots a DVO Curve (Desired Versus Observed) on Weights & Biases.
    Returns the corresponding Prompt Alignment value of the DVO plot.
    """
    assert desired_losses.dim() == observed_losses.dim() == 2
    assert desired_losses.size(0) == observed_losses.size(0)
    assert desired_losses.size(1) + 1 == observed_losses.size(1)
    num_models = desired_losses.size(0)
    keys = [f'dnn_{tag}_{i}' for i in range(num_models)] + ['identity']
    max_desired_loss = desired_losses.max().item()
    xs = desired_losses.tolist() + [[0, max_desired_loss]]
    ys = observed_losses[..., 1:].tolist() + [[0, max_desired_loss]]
    if optimal_loss is not None:  # Add a horizontal line indicating the optimal loss:
        xs = [[0, max_desired_loss]] + xs
        ys = [[optimal_loss, optimal_loss]] + ys
        keys = ['optimal'] + keys

    if filepath is not None:
        os.makedirs(f'figure_files/{filepath}', exist_ok=True)
        save_path = f'figure_files/{filepath}/dvo_{tag}_{title_postfix}.pt'
        torch.save({'x': xs, 'y': ys, 'key': keys}, save_path)

    table = lists2table(xs, ys, keys, xname='desired loss', yname='observed loss')
    plot_table = wandb.plot_table(
        "wandb/lineseries/v0",
        table,
        {"step": "desired loss", "lineKey": "model", "lineVal": "observed loss"},
        {
            "title": f"Observed Versus Desired Loss ({title_postfix})",
            "xname": "desired loss", "yname": "observed loss"
        }
    )
    wandb.log({
        f"observed_loss_vs_iters_{tag}_{title_postfix}, {tag}": plot_table
    })
    prompt_alignment = r_squared(preds=observed_losses[..., 1:], targets=desired_losses)
    wandb.log({
        f"Prompt Alignment ({tag}, {title_postfix})": prompt_alignment,
        "epoch": epoch
    })
    return prompt_alignment


def plot_time_versus_observed(
    observed_losses, tag='', title_postfix='', optimal_loss=None, gt_trajectories=None, filepath=None
):
    """
    Plots a Recursive Prompting Curve on Weights & Biases.
    """
    assert observed_losses.dim() == 2
    # TODO: For now, just a single trajectory is accepted
    assert gt_trajectories is None or gt_trajectories.dim() == 1
    num_models = observed_losses.size(0)
    num_iterations = observed_losses.size(1)
    xs = torch.arange(num_iterations).view(1, -1).repeat(num_models, 1)
    xs = xs.tolist()
    ys = observed_losses.tolist()
    keys = [f'dnn_{tag}_{i}' for i in range(num_models)]
    if optimal_loss is not None:  # Add a horizontal line indicating the optimal loss:
        xs.append([0, num_iterations])
        ys.append([optimal_loss, optimal_loss])
        keys.append('optimal')
    if gt_trajectories is not None:  # Overlay a training set trajectory from e.g., Adam, SGD, etc.
        xs.append(torch.arange(len(gt_trajectories)).tolist())
        ys.append(gt_trajectories.tolist())
        keys.append('gradient-based')

    if filepath is not None:
        os.makedirs(f'figure_files/{filepath}', exist_ok=True)
        save_path = f'figure_files/{filepath}/recursive_{tag}_{title_postfix}.pt'
        torch.save({'x': xs, 'y': ys, 'key': keys}, save_path)

    table = lists2table(xs, ys, keys, xname='iteration', yname='observed loss')
    plot_table = wandb.plot_table(
        "wandb/lineseries/v0",
        table,
        {"step": "iteration", "lineKey": "model", "lineVal": "observed loss"},
        {
            "title": f"Observed Loss Over Recursions ({title_postfix}, {tag})",
            "xname": "iteration", "yname": "observed loss"
        }
    )
    wandb.log({
        f"iterloss_{tag}_{title_postfix}": plot_table
    })


def moduleify(Gpt_output, net_constructor, unnormalize_fn):
    """
    Gpt_output: (N, D) tensor (N = batch_size, D = number of parameters)
    net_constructor: Function (should take no args/kwargs) that returns a randomly-initialized neural network
                     with the appropriate architecture
    unnormalize_fn: Function that takes a (N, D) tensor and "unnormalizes" it back to the original parameter space

    Returns: A length-N list of nn.Module instances, where the i-th nn.Module has the parameters from Gpt_output[i].
             If N = 1, then a single nn.Module is returned instead of a list.
    """
    Gpt_output = unnormalize_fn(Gpt_output)
    num_nets = Gpt_output.size(0)
    net_instance = net_constructor()
    target_state_dict = net_instance.state_dict()
    parameter_names = target_state_dict.keys()
    parameter_sizes = [v.size() for v in target_state_dict.values()]
    parameter_chunks = [v.numel() for v in target_state_dict.values()]

    parameters = torch.split(Gpt_output, parameter_chunks, dim=1)
    modules = []
    for i in range(num_nets):
        net = net_constructor()
        # Build a state dict from the generated parameters:
        state_dict = {
            pname: param[i].reshape(size) for pname, param, size in \
                zip(parameter_names, parameters, parameter_sizes)
        }
        net.load_state_dict(state_dict, strict=True)
        modules.append(net)
    if len(modules) == 1:  # don't return in list format if there's only one network
        modules = modules[0]
    return modules
