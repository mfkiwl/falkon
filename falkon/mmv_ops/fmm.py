from contextlib import ExitStack
from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.cuda as tcd

import falkon
from falkon.mmv_ops.utils import *
from falkon.options import BaseOptions
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.device_copy import copy
from falkon.utils.helpers import (
    sizeof_dtype, select_dim_over_nm, calc_gpu_block_sizes
)
from falkon.utils.tensor_helpers import create_same_stride, create_fortran


@dataclass(frozen=True)
class ArgsFmm():
    X1: Union[torch.Tensor, SparseTensor]
    X2: Union[torch.Tensor, SparseTensor]
    out: torch.Tensor
    kernel: 'falkon.kernels.Kernel'
    gpu_dtype: torch.dtype
    max_mem: float
    differentiable: bool
    num_streams: int = 1


def mm_run_starter(proc_idx, queue, device_id):
    a: ArgsFmm = queue.get()
    X1, X2, out = a.X1, a.X2, a.out
    kernel, computation_dtype = a.kernel, a.gpu_dtype
    differentiable = a.differentiable
    max_mem = a.max_mem
    # `dev` decides where computations are run (cuda or cpu)
    if device_id < 0:
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda:%d' % device_id)
    # decide ooc: if ooc, data and computation devices are different
    is_ooc = dev.type != X1.device.type
    change_dtype = computation_dtype != X1.dtype
    is_sparse = isinstance(X1, SparseTensor) and isinstance(X2, SparseTensor)

    avail_mem = max_mem / sizeof_dtype(computation_dtype)
    extra_mem = kernel.extra_mem()

    if differentiable:
        diff_coef_nm = 10
        assert not is_sparse, "Sparse + differentiable mmvs are not supported"
        n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1],
                                  coef_nd=extra_mem.get('nd', 0) + 1,
                                  coef_md=extra_mem.get('md', 0) + 1,
                                  coef_nm=(extra_mem.get('nm', 0) + 1) * diff_coef_nm,
                                  coef_n=extra_mem.get('n', 0),
                                  coef_m=extra_mem.get('m', 0),
                                  rest=extra_mem.get('d', 0),
                                  max_mem=avail_mem)
    elif is_sparse:
        if is_ooc or change_dtype:
            # coef_nm = 3 because: assume density of output == 1, then 2*nm are necessary for the
            # output stored as CSR, + 1 for the dense output.
            # also note that the chunk of matrix `m2` gets transposed then copied on device, hence
            # it needs 2 * D * M * density elements.
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1],
                                      coef_nd=2 * X1.density,
                                      coef_md=2 * X2.density,
                                      coef_nm=3,
                                      coef_n=0,
                                      coef_m=0,
                                      rest=0,
                                      max_mem=avail_mem)
        else:
            # No allocation will be performed by us. Only in-kernel stuff.
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1],
                                      coef_nd=0,
                                      coef_md=0,
                                      coef_nm=0,
                                      coef_n=0,
                                      coef_m=0,
                                      rest=0,
                                      max_mem=avail_mem)
    else:
        if is_ooc or change_dtype:
            # Need to allocate extra buffers for data-type change, or device change
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1],
                                      coef_nd=extra_mem.get('nd', 0) + 1,
                                      coef_md=extra_mem.get('md', 0) + 1,
                                      coef_nm=extra_mem.get('nm', 0) + 1,
                                      coef_n=extra_mem.get('n', 0),
                                      coef_m=extra_mem.get('m', 0),
                                      rest=extra_mem.get('d', 0),
                                      max_mem=avail_mem)
        else:
            # No allocation will be performed by us. Only in-kernel stuff.
            n, m = select_dim_over_nm(max_n=X1.shape[0], max_m=X2.shape[0], d=X1.shape[1],
                                      coef_nd=extra_mem.get('nd', 0),
                                      coef_md=extra_mem.get('md', 0),
                                      coef_nm=extra_mem.get('nm', 0),
                                      coef_n=extra_mem.get('n', 0),
                                      coef_m=extra_mem.get('m', 0),
                                      rest=extra_mem.get('d', 0),
                                      max_mem=avail_mem)

    # Run
    if differentiable:
        mm_diff_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev)
    elif is_sparse:
        sparse_mm_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev)
    else:
        mm_run_thread(X1, X2, out, kernel, n, m, computation_dtype, dev)


def sparse_mm_run_thread(m1: SparseTensor, m2: SparseTensor, out: torch.Tensor,
                         kernel, n: int, m: int, comp_dt: torch.dtype, dev: torch.device):
    is_ooc = dev.type != m1.device.type
    change_dtype = comp_dt != m1.dtype
    N, D = m1.shape
    M = m2.shape[0]

    """ Initialize extra buffers """
    has_gpu_bufs = is_ooc or change_dtype
    dev_nm = None
    if has_gpu_bufs:
        dev_nm = create_same_stride((n, m), out, comp_dt, dev)

    """ Run splitting along N, M """
    with ExitStack() as stack:
        stream = None
        if dev.type == 'cuda':
            stack.enter_context(tcd.device(dev))
            stream = tcd.current_stream(dev)
            stack.enter_context(tcd.stream(stream))

        for j in range(0, M, m):
            lenj = min(m, M - j)

            c_m2 = m2.narrow_rows(j, lenj).to(dtype=comp_dt)
            # On CUDA the second argument to apply (a Sparse*Sparse multiplication) must be
            # in CSR format, which is generally inefficient (given D might be large). On CPU this
            # is not necessary, so we avoid it.
            if dev.type == 'cuda':  # Note that CUDA-incore is not allowed to happen.
                c_dev_m2 = SparseTensor.from_scipy(
                    c_m2.transpose_csc().to_scipy().tocsr(copy=False)) \
                    .index_to_int() \
                    .to(device=dev, non_blocking=True)
            else:
                c_dev_m2 = c_m2.transpose_csc()

            for i in range(0, N, n):
                leni = min(n, N - i)

                c_m1 = m1.narrow_rows(i, leni).to(dtype=comp_dt)
                if dev.type == 'cuda':
                    c_dev_m1 = c_m1.index_to_int().to(device=dev, non_blocking=True)
                else:
                    c_dev_m1 = c_m1

                if has_gpu_bufs:
                    c_dev_out = dev_nm[:leni, :lenj]
                else:
                    c_dev_out = out[i: i + leni, j: j + lenj]
                c_dev_out.fill_(0.0)

                c_dev_out = kernel.compute_sparse(c_m1, c_m2, c_dev_out)
                ddd = kernel._prepare_sparse(c_m1, c_m2)
                c_dev_out = kernel._apply_sparse(c_dev_m1, c_dev_m2, c_dev_out)
                c_dev_out = kernel._finalize(c_dev_out, ddd)

                # Copy back to host
                if has_gpu_bufs:
                    copy(c_dev_out, out[i: i + leni, j: j + lenj], s=stream,
                         allow_dtype_change=True)


def mm_run_thread(m1: torch.Tensor, m2: torch.Tensor, out: torch.Tensor,
                  kernel, n: int, m: int, comp_dt: torch.dtype, dev: torch.device):
    is_ooc = dev.type != m1.device.type
    change_dtype = comp_dt != m1.dtype
    N, D = m1.shape
    M = m2.shape[0]

    """ Initialize extra buffers """
    flat_offset = 0
    total_memory = 0
    has_gpu_bufs = is_ooc or change_dtype
    if has_gpu_bufs:
        total_memory += n * m + n * D + m * D
    flat_dev_t = torch.empty(size=(total_memory,), dtype=comp_dt, device=dev)
    dev_nm, dev_m1, dev_m2 = None, None, None
    if has_gpu_bufs:
        dev_nm, flat_offset = _extract_flat(flat_dev_t, size=(n, m), other=out, offset=flat_offset)
        dev_m1, flat_offset = _extract_flat(flat_dev_t, size=(n, D), other=m1, offset=flat_offset)
        dev_m2, flat_offset = _extract_flat(flat_dev_t, size=(m, D), other=m2, offset=flat_offset)

    """ Run splitting along N, M """
    with ExitStack() as stack:
        stream = None
        if dev.type == 'cuda':
            stack.enter_context(tcd.device(dev))
            stream = tcd.current_stream(dev)
            stack.enter_context(tcd.stream(stream))

        for i in range(0, N, n):
            leni = min(n, N - i)

            if has_gpu_bufs:
                c_dev_m1 = copy(m1[i: i + leni, :], dev_m1[:leni, :], s=stream,
                                allow_dtype_change=True)
            else:
                c_dev_m1 = m1[i: i + leni, :]

            for j in range(0, M, m):
                lenj = min(m, M - j)

                if has_gpu_bufs:
                    c_dev_m2 = copy(m2[j: j + lenj, :], dev_m2[:lenj, :], s=stream,
                                    allow_dtype_change=True)
                    c_dev_out = dev_nm[:leni, :lenj]
                else:
                    c_dev_m2 = m2[j: j + lenj, :]
                    c_dev_out = out[i: i + leni, j: j + lenj]
                c_dev_out.fill_(0.0)

                # Compute kernel sub-matrix
                kernel.compute(c_dev_m1, c_dev_m2, c_dev_out)

                # Copy back to host
                if has_gpu_bufs:
                    copy(c_dev_out, out[i: i + leni, j: j + lenj], s=stream,
                         allow_dtype_change=True)


def mm_diff_run_thread(m1: torch.Tensor, m2: torch.Tensor, out: torch.Tensor,
                       kernel, n: int, m: int, comp_dt: torch.dtype, dev: torch.device):
    N, D = m1.shape
    M = m2.shape[0]

    """ Run splitting along N, M """
    if m == M:  # TODO: This makes no sense, but prevents a AutoGrad error from occuring
        m = m - 1
    with ExitStack() as stack:
        if dev.type == 'cuda':
            stack.enter_context(tcd.device(dev))
            stream = tcd.current_stream(dev)
            stack.enter_context(tcd.stream(stream))

        for i in range(0, N, n):
            leni = min(n, N - i)
            c_dev_m1 = m1[i: i + leni, :].to(device=dev, dtype=comp_dt, non_blocking=True,
                                             copy=False)
            for j in range(0, M, m):
                lenj = min(m, M - j)
                c_dev_m2 = m2[j: j + lenj, :].to(device=dev, dtype=comp_dt, non_blocking=True,
                                                 copy=False)
                c_dev_out = kernel.compute_diff(c_dev_m1, c_dev_m2)
                out[i: i + leni, j: j + lenj] = c_dev_out.to(
                    device=out.device, dtype=out.dtype, non_blocking=False, copy=False)


def fmm(X1: Union[torch.Tensor, SparseTensor],
        X2: Union[torch.Tensor, SparseTensor],
        kernel: 'falkon.kernels.Kernel',
        out: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        opt: Optional[BaseOptions] = None) -> torch.Tensor:
    """
    performs fnc(X1*X2', X1, X2) in blocks on multiple GPUs
    """
    is_sparse = isinstance(X1, SparseTensor)

    if not is_sparse:
        _check_contiguity((X1, 'X1'), (X2, 'X2'), (out, 'out'))

    N = X1.shape[0]
    M = X2.shape[0]
    data_dev = X1.device
    comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'
    if out is None:
        if is_sparse:
            out = create_fortran((N, M), dtype=X1.dtype, device=data_dev,
                                 pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')
        else:
            out = create_same_stride((N, M), X1, X1.dtype, device=data_dev,
                                     pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')

    # If float32 we need to upcast to float64 to avoid numerical precision errors in the kernel
    comp_dtype = X1.dtype
    if sizeof_dtype(comp_dtype) < 8 and opt.no_single_kernel:
        comp_dtype = torch.float64

    if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
        args = ArgsFmm(X1=X1, X2=X2, out=out, kernel=kernel, gpu_dtype=comp_dtype,
                       max_mem=opt.max_cpu_mem, num_streams=1, differentiable=differentiable)
        _call_direct(mm_run_starter, (args, -1))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
        if is_sparse:
            raise NotImplementedError("In-core, sparse fmm not implemented. "
                                      "Use the out-of-core version instead.")
        gpu_info = _get_gpu_info(opt, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmm(X1=X1, X2=X2, out=out, kernel=kernel, gpu_dtype=comp_dtype,
                       max_mem=single_gpu_info.usable_memory, num_streams=opt.num_fmm_streams,
                       differentiable=differentiable)
        _call_direct(mm_run_starter, (args, data_dev.index))
    elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
        gpu_info = _get_gpu_info(opt, slack=0.9)
        args = []  # Arguments passed to each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, N)
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            if is_sparse:
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFmm(X1=X1_block, X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
                                 kernel=kernel, gpu_dtype=comp_dtype, max_mem=g.usable_memory,
                                 num_streams=opt.num_fmm_streams, differentiable=differentiable),
                         g.Id))
        _start_wait_processes(mm_run_starter, args)
    else:
        raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
    return out


# noinspection PyMethodOverriding
class KernelMmFnFull(torch.autograd.Function):
    @staticmethod
    def run_cpu_cpu(X1, X2, out, kernel, dtype, options, diff):
        args = ArgsFmm(X1=X1, X2=X2, out=out, kernel=kernel, gpu_dtype=dtype,
                       max_mem=options.max_cpu_mem, num_streams=1, differentiable=diff)
        _call_direct(mm_run_starter, (args, -1))
        return out

    @staticmethod
    def run_cpu_gpu(X1, X2, out, kernel, dtype, options, diff):
        gpu_info = _get_gpu_info(options, slack=0.9)
        args = []  # Arguments passed to each subprocess
        block_sizes = calc_gpu_block_sizes(gpu_info, X1.shape[0])
        for i, g in enumerate(gpu_info):
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            if isinstance(X1, SparseTensor):
                X1_block = X1.narrow_rows(block_sizes[i], bwidth)
            else:
                X1_block = X1.narrow(0, block_sizes[i], bwidth)
            args.append((ArgsFmm(X1=X1_block, X2=X2, out=out.narrow(0, block_sizes[i], bwidth),
                                 kernel=kernel, gpu_dtype=dtype, max_mem=g.usable_memory,
                                 num_streams=options.num_fmm_streams, differentiable=diff),
                         g.Id))
        _start_wait_processes(mm_run_starter, args)
        return out

    @staticmethod
    def run_gpu_gpu(X1, X2, out, kernel, dtype, options, diff):
        if isinstance(X1, SparseTensor):
            raise NotImplementedError("In-core, sparse fmm not implemented. "
                                      "Use the out-of-core version instead.")
        data_dev = X1.device
        gpu_info = _get_gpu_info(options, slack=0.9)
        single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
        args = ArgsFmm(X1=X1, X2=X2, out=out, kernel=kernel, gpu_dtype=dtype,
                       max_mem=single_gpu_info.usable_memory, num_streams=options.num_fmm_streams,
                       differentiable=diff)
        _call_direct(mm_run_starter, (args, data_dev.index))
        return out

    @staticmethod
    def forward(ctx,
                kernel: 'falkon.kernels.Kernel',
                opt: Optional[BaseOptions],
                out: Optional[torch.Tensor],
                X1: Union[torch.Tensor, SparseTensor],
                X2: Union[torch.Tensor, SparseTensor],
                *kernel_params,
                ):
        is_sparse = isinstance(X1, SparseTensor)
        if is_sparse:
            differentiable = False
        else:
            _check_contiguity((X1, 'X1'), (X2, 'X2'), (out, 'out'))
            differentiable = any([t.requires_grad for t in [X1, X2] + list(*kernel_params)])

        N = X1.shape[0]
        M = X2.shape[0]
        data_dev = X1.device
        comp_dev_type = 'cpu' if opt.use_cpu or not torch.cuda.is_available() else 'cuda'
        if out is None:
            if is_sparse:
                out = create_fortran((N, M), dtype=X1.dtype, device=data_dev,
                                     pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')
            else:
                out = create_same_stride((N, M), X1, X1.dtype, device=data_dev,
                                         pin_memory=data_dev.type != 'cuda' and comp_dev_type == 'cuda')
        # If float32 we need to upcast to float64 to avoid numerical precision errors in the kernel
        comp_dtype = X1.dtype
        if sizeof_dtype(comp_dtype) < 8 and opt.no_single_kernel:
            comp_dtype = torch.float64

        with torch.inference_mode():
            X1d = X1.detach()
            X2d = X2.detach()
            kerneld = kernel.detach()
            if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
                out = KernelMmFnFull.run_cpu_cpu(X1d, X2d, out, kerneld, comp_dtype, opt, False)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
                out = KernelMmFnFull.run_gpu_gpu(X1d, X2d, out, kerneld, comp_dtype, opt, False)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
                out = KernelMmFnFull.run_cpu_gpu(X1d, X2d, out, kerneld, comp_dtype, opt, False)
            else:
                raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
        if not differentiable:
            ctx.mark_non_differentiable(out)
        else:
            ctx.save_for_backward(X1, X2, *kernel_params)
            ctx.out = out
            ctx.kernel = kernel
            ctx.opt = opt
            ctx.comp_dtype = comp_dtype
        return out

    @staticmethod
    def backward(ctx, outputs):
        X1, X2, *kernel_params = ctx.saved_tensors

        data_dev = X1.device
        comp_dev_type = 'cpu' if ctx.opt.use_cpu or not torch.cuda.is_available() else 'cuda'

        # We must rerun MM in differentiable mode this time.
        with torch.autograd.enable_grad():
            if comp_dev_type == 'cpu' and data_dev.type == 'cpu':
                out = KernelMmFnFull.run_cpu_cpu(X1, X2, ctx.out.detach(), ctx.kernel, ctx.comp_dtype, ctx.opt, True)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cuda':
                out = KernelMmFnFull.run_gpu_gpu(X1, X2, ctx.out.detach(), ctx.kernel, ctx.comp_dtype, ctx.opt, True)
            elif comp_dev_type == 'cuda' and data_dev.type == 'cpu':
                out = KernelMmFnFull.run_cpu_gpu(X1, X2, ctx.out.detach(), ctx.kernel, ctx.comp_dtype, ctx.opt, True)
            else:
                raise RuntimeError("Requested CPU computations with CUDA data. This should not happen.")
            bwd = (out * outputs).sum()

        saved_idx = 0
        needs_grad = []
        for i, i_grad in enumerate(ctx.needs_input_grad):
            if i_grad:
                needs_grad.append(ctx.saved_tensors[saved_idx])
                saved_idx += 1
        grads = torch.autograd.grad(
            bwd, needs_grad, retain_graph=False, allow_unused=False)
        grads_idx = 0
        results = []
        for i, i_grad in enumerate(ctx.needs_input_grad):
            if i_grad:
                results.append(grads[grads_idx])
                grads_idx += 1
            else:
                results.append(None)
        return tuple(results)
