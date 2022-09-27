"""
This script filters raw checkpoints to remove those with NaN/inf/large weights.
It also estimates the variance of parameters in order to pre-process them for G.pt training.
"""
import lmdb

try:
    import isaacgym
except ImportError:
    print("WARNING: Isaac Gym not imported")

import torch
from tqdm import tqdm
from Gpt.data.dataset_lmdb import ParameterDataset
from Gpt.diffusion import create_diffusion
import shutil


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Example using Cartpole:
    dataset_name = "cartpole"
    dataset_dir = "checkpoint_datasets/cartpole"
    train_metric = "train_ret"
    num_test_runs = 2050

    check_for_bad_runs = True
    compute_variance = True
    compute_diffusion_prior = True

    if check_for_bad_runs:

        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=0,
            normalizer_name="none"
        )

        bad_runs = []
        for run in tqdm(range(dataset.num_runs), total=dataset.num_runs):
            #print("Run = {}".format(dataset.runs[run]))
            for iters in range(200):
                try:
                    net = dataset.get_run_network(run, iters)
                except (lmdb.CorruptedError, lmdb.PageNotFoundError):
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
                    continue
                if not isinstance(net, (torch.FloatTensor, torch.Tensor)):
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
                    continue
                big_weights = (net.abs().amax() > 10).item()
                illegal_weights = torch.isfinite(net).all().logical_not_().item()
                if big_weights or illegal_weights:
                    print(f'Bad Run Found (iter={iters}): {dataset.runs[run]}')
                    bad_runs.append(dataset.runs[run])
        bad_runs = set(bad_runs)
        print(f"Deleting following bad runs: {list(bad_runs)}")
        for bad_run in bad_runs:
            shutil.rmtree(f"{dataset_dir}/{bad_run}")
        print('Done checking for bad runs.')

        del dataset

    if compute_variance:

        dataset = ParameterDataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            train_metric=train_metric,
            num_test_runs=num_test_runs,
            normalizer_name="none",
        )

        num_checkpoints_per_run = 200
        num = 25000
        torch.manual_seed(10)
        rand_runs = torch.randint(low=0, high=dataset.num_runs, size=(num,)).tolist()
        rand_iters = torch.randint(low=0, high=num_checkpoints_per_run, size=(num,)).tolist()
        runs_and_iters = zip(rand_runs, rand_iters)
        nets = [dataset.get_run_network(run, iteration) for run, iteration in tqdm(runs_and_iters, total=num)]

        nets = torch.stack(nets)
        stdev = nets.flatten().std(unbiased=True).item()
        oai_coeff = 0.538 / stdev   # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
        print(f'Standard Deviation: {stdev}')
        print(f'OpenAI Coefficient: {oai_coeff}')

        if compute_diffusion_prior:
            diffusion = create_diffusion(
                learn_sigma=False, predict_xstart=True,
                noise_schedule='linear', steps=1000
            )
            prior_kl = diffusion._prior_bpd(nets.cuda() * oai_coeff)
            print(f'Prior KL: {prior_kl.mean().item()}')
