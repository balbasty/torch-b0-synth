import torch
from torch import Tensor
from typing import List


def identity_grid(shape, dtype=None, device=None):
    """Returns an identity deformation field.

    Parameters
    ----------
    shape : (dim,) sequence of int
        Spatial dimension of the field.
    dtype : torch.dtype, default=`get_default_dtype()`
        Data type.
    device torch.device, optional
        Device.

    Returns
    -------
    grid : (*shape, dim) tensor
        Transformation field

    """
    mesh1d = [torch.arange(float(s), dtype=dtype, device=device)
              for s in shape]
    grid = meshgrid_ij(*mesh1d)
    grid = torch.stack(grid, dim=-1)
    return grid


def prod(sequence):
    """Perform the cumulative product of a sequence of elements.
    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    Returns
    -------
    product : T
        Product of the elements in the sequence.
    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
    return accumulate


def torch_version(mode, version):
    """Check torch version

    Parameters
    ----------
    mode : {'<', '<=', '>', '>='}
    version : tuple[int]

    Returns
    -------
    True if "torch.version <mode> version"

    """
    current_version, *cuda_variant = torch.__version__.split('+')
    major, minor, patch, *_ = current_version.split('.')
    # strip alpha tags
    for x in 'abcdefghijklmnopqrstuvwxy':
        if x in patch:
            patch = patch[:patch.index(x)]
    current_version = (int(major), int(minor), int(patch))
    return _compare_versions(current_version, mode, list(version))


def _compare_versions(version1, mode, version2):
    for v1, v2 in zip(version1, version2):
        if mode in ('gt', '>'):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('ge', '>='):
            if v1 > v2:
                return True
            elif v1 < v2:
                return False
        elif mode in ('lt', '<'):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
        elif mode in ('le', '<='):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False
    if mode in ('gt', 'lt', '>', '<'):
        return False
    else:
        return True


if torch_version('>=', (1, 10)):
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x, indexing='ij')
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x, indexing='xy')
    meshgrid_ij = lambda *x: torch.meshgrid(*x, indexing='ij')
    meshgrid_xy = lambda *x: torch.meshgrid(*x, indexing='xy')
else:
    @torch.jit.script
    def meshgrid_script_ij(x: List[torch.Tensor]) -> List[Tensor]:
        return torch.meshgrid(x)
    @torch.jit.script
    def meshgrid_script_xy(x: List[torch.Tensor]) -> List[Tensor]:
        grid = torch.meshgrid(x)
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid
    meshgrid_ij = lambda *x: torch.meshgrid(*x)
    def meshgrid_xy(*x):
        grid = list(torch.meshgrid(*x))
        if len(grid) > 1:
            grid[0] = grid[0].transpose(0, 1)
            grid[1] = grid[1].transpose(0, 1)
        return grid
