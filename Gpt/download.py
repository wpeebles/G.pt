"""
Functions for downloading pre-trained G.pt models and checkpoint datasets.
"""
from torchvision.datasets.utils import download_and_extract_archive, download_url
import torch
import os
from Gpt.distributed import is_main_proc, synchronize


# These are the pre-trained G.pt checkpoints we currently have available for download:
pretrained_models = {'mnist_loss.pt', 'mnist_error.pt', 'cartpole.pt', 'cifar10_loss.pt', 'cifar10_error.pt'}
# These are the different per-task checkpoint datasets we have available for download:
checkpoint_datasets = {'mnist', 'cartpole', 'cifar10'}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)
    else:  # Load a custom G.pt checkpoint:
        assert os.path.isfile(model_name), f'Could not find G.pt checkpoint at {model_name}'
        return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained G.pt model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path) and is_main_proc():  # download (only on primary process)
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'http://learn2learn.eecs.berkeley.edu/pretrained_models/{model_name}'
        download_url(web_path, 'pretrained_models')
    synchronize()  # Wait for the primary process to finish downloading the checkpoint
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def find_checkpoints(dataset_path):
    """
    Finds a checkpoint dataset, downloading it if necessary. Alternatively, loads a dataset from a local path.
    """
    dataset_name = process_dset_name(dataset_path)
    if dataset_name in checkpoint_datasets:
        return download_individual_checkpoint_dataset(dataset_path)
    else:
        assert os.path.isdir(dataset_path), f'Could not find a checkpoint dataset at {dataset_path}'
        return dataset_path


def download_individual_checkpoint_dataset(dataset_path):
    """
    Downloads a checkpoint dataset from the web.
    """
    dataset_name = process_dset_name(dataset_path)
    dataset_dir = os.path.dirname(dataset_path)
    if not os.path.isdir(dataset_path) and is_main_proc():
        os.makedirs(dataset_dir, exist_ok=True)
        web_path = f'http://learn2learn.eecs.berkeley.edu/checkpoint_datasets/{dataset_name}.zip'
        download_and_extract_archive(web_path, dataset_dir, remove_finished=True)
    else:
        print(f'Found pre-existing {dataset_path} directory')
    synchronize()  # Wait for the primary process to finish downloading the dataset
    return dataset_path


def process_dset_name(dataset_name):
    """
    Processes a dataset name to make sure it's in the correct format.
    """
    dataset_name = os.path.basename(dataset_name).split('_')[0].lower()
    assert dataset_name in checkpoint_datasets
    return dataset_name


if __name__ == "__main__":
    # Download all three checkpoint datasets and all five pre-trained G.pt models
    dataset_dir = "checkpoint_datasets"
    for dataset in checkpoint_datasets:
        find_checkpoints(os.path.join(dataset_dir, dataset))
    for model_name in pretrained_models:
        find_model(model_name)
    print('Checkpoint datasets available in checkpoint_datasets directory')
    print('Pre-trained G.pt models available in pretrained_models directory')
    print('Finished.')
