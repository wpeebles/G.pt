"""
Various helper functions for permutation augmentation of neural networks.
"""
import torch


def permute_out(tensors, permutation):
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    for tensor in tensors:
        tensor.copy_(tensor[permutation])


def permute_in(tensors, permutation):
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    for tensor in tensors:
        tensor.copy_(tensor[:, permutation])


def permute_in_out(tensor, permutation_in, permutation_out):
    tensor.copy_(tensor[:, permutation_in][permutation_out])


def random_permute_flat(nets, architecture, seed, permutation_fn):

    """
    Applies an output-preserving parameter permutation to a list of nets, eah with shape (D,).
    The same permutation is applied to each network in the list.

    nets: list of torch.Tensor or torch.Tensor, each of shape (D,)
    """

    input_is_tensor = isinstance(nets, torch.Tensor)
    if input_is_tensor:
        nets = [nets]
    full_permute = []
    total = 0

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    def build_index(tensors, permutation, permutation2=None, permute=None):
        nonlocal total
        if isinstance(tensors, torch.Tensor):
            tensors = (tensors,)
        for tensor in tensors:
            num_params = tensor.numel()
            shape = tensor.shape
            indices = torch.arange(total, num_params + total).view(shape)
            if permute == 'out':
                indices = indices[permutation]
            elif permute == 'in':
                indices = indices[:, permutation]
            elif permute == 'both':
                indices = indices[:, permutation][permutation2]
            elif permute == 'none':
                pass
            total += num_params
            full_permute.append(indices.flatten())

    build_in_fn = lambda *args: build_index(*args, permute='in')
    build_out_fn = lambda *args: build_index(*args, permute='out')
    build_both_fn = lambda *args: build_index(*args, permute='both')
    register_fn = lambda x: build_index(x, permutation=None, permute='none')
    permutation_fn(architecture, generator, build_in_fn, build_out_fn, build_both_fn, register_fn)

    full_permute = torch.cat(full_permute)
    assert total == full_permute.size(0) == nets[0].size(0)
    permuted_nets = [net[full_permute] for net in nets]  # Apply the same permutation to each net
    if input_is_tensor:  # Unpack the list to return in same format as input
        permuted_nets = permuted_nets[0]
    return permuted_nets
