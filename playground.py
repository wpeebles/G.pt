"""
A simple script that loads and runs pre-trained G.pt models (more minimalist than main.py).
"""
import isaacgym
import torch
import numpy as np
from Gpt.models.transformer import Gpt
from Gpt.data.dataset_lmdb import ParameterDataset
from Gpt.diffusion import create_diffusion
from Gpt.distributed import get_world_size, get_rank
from Gpt.tasks import get
from Gpt.latent_walk_helpers import create_latent_walk_for_cnn, slerpify
from Gpt.vis import synth
from Gpt.download import find_model
import hydra
import random
from tqdm import tqdm


def set_seed(cfg):
    seed = cfg.rng_seed * get_world_size() + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    return seed


def ddim_synth(
    diffusion,
    G,
    loss_target,        # The prompted loss/error/return: shape (N, 1)
    loss_prev,          # The starting loss/error/return: shape (N, 1)
    w_prev,             # The starting parameter vector: shape (N, D)
    **ddim_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process using DDIM sampling.
    Specifically, this function draws a sample from p(theta^*|prompt_loss,starting_loss,starting_theta).
    """
    assert loss_target.size(0) == loss_prev.size(0) == w_prev.size(0)

    model_kwargs = {
        'loss_target': loss_target,
        'loss_prev': loss_prev,
        'x_prev': w_prev
    }

    shape = w_prev.shape
    sample = diffusion.ddim_sample_loop(
        G,
        shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device='cuda',
        **ddim_sample_loop_kwargs
    )

    return sample


def generate_interpolation_samples(model, diffusion, dataset, seed=0):
    """
    Creates an MP4 showing a latent walk through parameter space. Most of this function is general,
    but the create_latent_walk_for_cnn function is specific to the CIFAR-10 CNN architecture.
    """
    assert "cifar10" in dataset.dataset_name, "Currently, only CIFAR-10 G.pt models support latent walks."
    # --------------------------
    n_videos = 1
    n_samples = 16
    n_steps = 120
    use_fixed_input_theta = True
    # --------------------------
    n_total_samples = n_videos * n_samples
    run_indices = list(range(seed * n_total_samples, (seed + 1) * n_total_samples))
    dim = dataset.get_run_network(run_index=0, iter=0, normalize=True, augment=False).size(0)
    noise = slerpify(torch.randn(n_videos, n_samples, dim, device="cuda"), n_steps)  # Create looping noise
    nets, lps = [], []  # nets = input (starting) parameters, lps = starting losses/errors
    for run_index in tqdm(run_indices):
        net = dataset.get_run_network(run_index=run_index, iter=0, normalize=True, augment=False).unsqueeze(0).cuda()
        lp = dataset.get_run_losses(run_index=run_index)[0].view(1).to("cuda")
        nets.append(net)
        lps.append(lp)
    nets = slerpify(torch.cat(nets, 0).view(n_videos, n_samples, dim), n_steps)  # (n_videos, n_samples, n_steps, D)
    if use_fixed_input_theta:
        # Use the same starting parameters for every frame in the video:
        nets = nets[0, 0, 0].view(1, 1, 1, -1).repeat(n_videos, n_samples, n_steps, 1)  # (n_videos, n_samples, n_steps, D)
    # Use a constant starting loss/error to better isolate the effect of sampling noise:
    lps = torch.cat(lps, 0).mean().view(1, 1, 1, 1).repeat(n_videos, n_samples, n_steps, 1)  # (n_videos, n_samples, n_steps, 1)
    # lts are the loss/error prompts (also constant):
    lts = torch.tensor(get(dataset.dataset_name, "best_prompt"), device="cuda").view(1, 1, 1, 1).repeat(n_videos, n_samples, n_steps, 1)  # (n_videos, n_samples, n_steps, 1)
    # Sample the updated parameters:
    preds = ddim_synth(diffusion, model, lts.view(-1, 1), lps.view(-1, 1), nets.view(-1, dim), noise=noise.view(-1, dim), progress=True)
    preds = dataset.unnormalize(preds)  # (N, D)
    preds = preds.reshape(n_videos, n_samples, n_steps, dim).permute(1, 2, 0, 3).cpu()
    # Generate and save the video:
    create_latent_walk_for_cnn(preds, filename="walk.mp4")


def playground(cfg):

    set_seed(cfg)
    state_dict = find_model(cfg.resume_path)

    train_dataset = ParameterDataset(
        dataset_dir=cfg.dataset.path,
        dataset_name=cfg.dataset.name,
        num_test_runs=cfg.dataset.num_test_runs,
        openai_coeff=cfg.dataset.openai_coeff,
        normalizer_name=cfg.dataset.normalizer,
        split="train",
        train_metric=cfg.dataset.train_metric,
        permute_augment=cfg.dataset.augment,
        target_epoch_size=cfg.dataset.target_epoch_size
    )

    dataset = ParameterDataset(
        dataset_dir=cfg.dataset.path,
        dataset_name=cfg.dataset.name,
        num_test_runs=cfg.dataset.num_test_runs,
        openai_coeff=cfg.dataset.openai_coeff,
        normalizer_name=cfg.dataset.normalizer,
        min_val=train_dataset.min_val,
        max_val=train_dataset.max_val,
        split="test",
        train_metric=cfg.dataset.train_metric,
        permute_augment=cfg.dataset.augment,
        target_epoch_size=cfg.dataset.target_epoch_size
    )

    model = Gpt(
        parameter_sizes=train_dataset.parameter_sizes,
        parameter_names=train_dataset.parameter_names,
        predict_xstart=cfg.transformer.predict_xstart,
        absolute_loss_conditioning=cfg.transformer.absolute_loss_conditioning,
        chunk_size=cfg.transformer.chunk_size,
        split_policy=cfg.transformer.split_policy,
        max_freq_log2=cfg.transformer.max_freq_log2,
        num_frequencies=cfg.transformer.num_frequencies,
        n_embd=cfg.transformer.n_embd,
        encoder_depth=cfg.transformer.encoder_depth,
        decoder_depth=cfg.transformer.decoder_depth,
        n_layer=cfg.transformer.n_layer,
        n_head=cfg.transformer.n_head,
        attn_pdrop=cfg.transformer.dropout_prob,
        resid_pdrop=cfg.transformer.dropout_prob,
        embd_pdrop=cfg.transformer.dropout_prob
    )

    diffusion = create_diffusion(
        learn_sigma=False, predict_xstart=cfg.transformer.predict_xstart,
        noise_schedule='linear', steps=1000
    )

    if cfg.transformer.ema:
        print("Loading EMA model...")
        model.load_state_dict(state_dict["G_ema"])
    else:
        print("Loading instantaneous model...")
        model.load_state_dict(state_dict["G"])

    model = model.to("cuda")
    model.eval()

    generate_interpolation_samples(model, diffusion, dataset, seed=cfg.rng_seed)


def single_proc_playground(local_rank, port, world_size, cfg):
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:{}".format(port),
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    playground(cfg)
    torch.distributed.destroy_process_group()


@hydra.main(config_path="configs/test", config_name="config.yaml")
def main(cfg):

    torch.set_grad_enabled(False)

    if cfg.num_gpus > 1:
        # Select a port for proc group init randomly
        port_range = [10000, 65000]
        port = random.randint(port_range[0], port_range[1])
        # Start a process per GPU
        torch.multiprocessing.start_processes(
            playground,
            args=(port, cfg.num_gpus, cfg),
            nprocs=cfg.num_gpus,
            start_method="spawn"
        )
    else:
        playground(cfg)


if __name__ == "__main__":
    main()
