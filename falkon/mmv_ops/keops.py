import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch

from falkon.utils.stream_utils import sync_current_stream

from falkon.mmv_ops.utils import _get_gpu_info, create_output_mat, _start_wait_processes
from falkon.options import FalkonOptions, BaseOptions
from falkon.utils import decide_cuda
from falkon.utils.helpers import sizeof_dtype, calc_gpu_block_sizes, check_same_device
from pykeops.torch import Genred


@dataclass(frozen=True)
class ArgsFmmv:
    X1: torch.Tensor
    X2: torch.Tensor
    v: torch.Tensor
    other_vars: List[torch.Tensor]
    out: torch.Tensor
    gpu_ram: float
    backend: str
    function: callable


def _keops_dtype(dtype: torch.dtype) -> str:
    """Returns a string which represents the given data type.

    The string representation is necessary for KeOps which doesn't
    like type objects.
    """
    if dtype == torch.float64:
        return 'float64'
    elif dtype == torch.float32:
        return 'float32'
    else:
        raise NotImplementedError("Data type %s not recognized." % (dtype))


def _decide_backend(opt: BaseOptions, num_dim: int) -> str:
    """Switch between CPU and GPU backend for KeOps
    """
    if not decide_cuda(opt):
        return 'CPU'
    else:
        return 'GPU_1D'


def _estimate_split(N, M, D, T, R, ds):
    """Estimate the splits along dimensions N and M for a MVM to fit in memory

    The operations consist of computing the product between a kernel
    matrix (from a N*D and a M*D matrix) and a 'vector' of shape M*T
    This typically requires storage of the input and output matrices,
    which occupies (M + N)*(D + T) memory locations plus some intermediate
    buffers to perform computations.
    TODO: It is not clear how much intermediate memory KeOps requires;
    the only thing that is certain is that it is quadratic in D.
    For now we sidestep this issue by using a smaller R than what is
    actually available in GPU memory.

    This function calculates the split along N and M into blocks of size n*m
    so that we can compute the kernel-vector product between such blocks
    and still fit in GPU memory.

    Parameters
    -----------
     - N : int
        The first dimension of the kernel matrix
     - M : int
        The second dimension of the kernel matrix
     - D : int
        The data dimensionality
     - T : int
        The number of output columns
     - R : float
        The amount of memory available (in bytes)
     - ds : int
        The size in bytes of each element in the data matrices
        (e.g. 4 if the data is in single precision).

    Returns
    --------
     - n : int
        The block size to be used along the first dimension
     - m : int
        The block size along the second dimension of the kernel
        matrix

    Raises
    -------
    RuntimeError
        If the available memory `R` is insufficient to store even the smallest
        possible input matrices. This may happen if `D` is very large since we
        do not perform any splitting along `D`.

    Notes
    ------
    We find 'good' values of M, N such that
    N*(D+T) + M*(D+T) <= R/ds
    """
    R = R / ds

    # We have a linear equation in two variables (N, M)
    slope = -1
    intercept = R / (D + T)

    slack_points = 10
    # We try to pick a point at the edges such that only one kind of split
    # is necessary
    if N < intercept - 1:
        M = min(M, intercept + slope * N)
    elif M < intercept - 1:
        N = min(N, intercept + slope * M)
    else:
        # All points on the slope such that N, M > 0 are possible
        N = intercept - slack_points - 1
        M = intercept + slope * N

    if N <= 0 or M <= 0:
        raise RuntimeError(
            "Insufficient available GPU "
            "memory (available %.2fGB)" % (R * ds / 2 ** 30))

    return int(N), int(M)


def _single_gpu_method(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    backend = a.backend
    X1 = a.X1
    X2 = a.X2
    v = a.v
    oout = a.out
    other_vars = a.other_vars
    fn = a.function
    R = a.gpu_ram

    N, D = X1.shape
    M = X2.shape[0]
    T = v.shape[1]
    device = torch.device(f"cuda:{device_id}")

    # Second round of subdivision (only if necessary due to RAM constraints)
    n, m = _estimate_split(N, M, D, T, R, sizeof_dtype(X1.dtype))

    other_vars_dev = [ov.to(device, copy=False) for ov in other_vars]
    out_ic = oout.device.index == device_id

    # Process the two rounds of splitting with a nested loop.
    with torch.cuda.device(device_id):
        for mi in range(0, M, m):
            ml = min(m, M - mi)
            if ml != M and mi > 0:  # Then we must create a temporary output array
                out = torch.empty_like(oout)
            else:
                out = oout

            cX2 = X2[mi:mi + ml, :].to(device, copy=False)
            cv = v[mi:mi + ml, :].to(device, copy=False)

            for ni in range(0, N, n):
                nl = min(n, N - ni)
                cX1 = X1[ni:ni + nl, :].to(device, copy=False)
                cout = out[ni: ni + nl, :].to(device, copy=False)

                variables = [cX1, cX2, cv] + other_vars_dev
                fn(*variables, out=cout, device_id=device_id, backend=backend)
                if not out_ic:
                    out[ni: ni + nl, :].copy_(cout)
            if ml != M and mi > 0:
                oout.add_(out)

    return oout


def run_keops_mmv(X1: torch.Tensor,
                  X2: torch.Tensor,
                  v: torch.Tensor,
                  other_vars: List[torch.Tensor],
                  out: Optional[torch.Tensor],
                  formula: str,
                  aliases: List[str],
                  axis: int,
                  reduction: str = 'Sum',
                  differentiable: bool = False,
                  opt: Optional[FalkonOptions] = None) -> torch.Tensor:
    if opt is None:
        opt = FalkonOptions()
    # Choose backend
    N, D = X1.shape
    T = v.shape[1]
    backend = _decide_backend(opt, D)
    dtype = _keops_dtype(X1.dtype)
    data_devs = [X1.device, X2.device, v.device]

    if any([ddev.type == 'cuda' for ddev in data_devs]) and (not backend.startswith("GPU")):
        warnings.warn("KeOps backend was chosen to be CPU, but GPU input tensors found. "
                      "Defaulting to 'GPU_1D' backend. To force usage of the CPU backend, "
                      "please pass CPU tensors; to avoid this warning if the GPU backend is "
                      "desired, check your options (i.e. set 'use_cpu=False').")
        backend = "GPU_1D"

    if differentiable:
        from falkon.kernels.tiling_red import TilingGenred
        fn = TilingGenred(formula, aliases, reduction_op='Sum', axis=1, dtype=dtype,
                          dtype_acc="auto", sum_scheme="auto", opt=opt)
        return fn(X1, X2, v, *other_vars, out=out, backend=backend)

    # Define formula wrapper
    fn = Genred(formula, aliases,
                reduction_op=reduction, axis=axis,
                dtype=dtype, dtype_acc=opt.keops_acc_dtype,
                sum_scheme=opt.keops_sum_scheme)

    comp_dev_type = backend[:3].lower().replace('gpu', 'cuda')  # 'cpu' or 'cuda'
    out = create_output_mat(out, data_devs, is_sparse=False, shape=(N, T), dtype=X1.dtype,
                            comp_dev_type=comp_dev_type, other_mat=X1, output_stride="C")

    if comp_dev_type == 'cpu' and all([ddev.type == 'cpu' for ddev in data_devs]):  # incore CPU
        variables = [X1, X2, v] + other_vars
        out = fn(*variables, out=out, backend=backend)
    elif comp_dev_type == 'cuda' and all([ddev.type == 'cuda' for ddev in data_devs]):  # incore CUDA
        variables = [X1, X2, v] + other_vars
        device = data_devs[0]
        with torch.cuda.device(device):
            sync_current_stream(device)
            out = fn(*variables, out=out, backend=backend)
    else:  # Out of core
        # slack is high due to imprecise memory usage estimates for keops
        gpu_info = _get_gpu_info(opt, slack=opt.keops_memory_slack)
        block_sizes = calc_gpu_block_sizes(gpu_info, N)

        # Create queues
        args = []  # Arguments passed to each subprocess
        for i, g in enumerate(gpu_info):
            # First round of subdivision
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            args.append((ArgsFmmv(
                X1=X1.narrow(0, block_sizes[i], bwidth),
                X2=X2,
                v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                other_vars=other_vars,
                function=fn,
                backend=backend,
                gpu_ram=g.usable_memory
            ), g.Id))
        _start_wait_processes(_single_gpu_method, args)

    return out
