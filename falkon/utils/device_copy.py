import torch.cuda

from .helpers import sizeof_dtype
from .tensor_helpers import is_f_contig, is_contig
if torch.cuda.is_available():
    from falkon.cuda.cudart_gpu import cuda_memcpy2d, cuda_memcpy2d_async
    from falkon.cuda.cublas_gpu import (cublasSetMatrix, cublasSetMatrixAsync,
                                        cublasGetMatrix, cublasGetMatrixAsync)


def check_copy(origin, dest, check_dtypes=True):
    if check_dtypes:
        # Data-types
        if origin.dtype != dest.dtype:
            raise ValueError("Data types of origin and destination (%s, %s) do not match." % (origin.dtype, dest.dtype))
    # Sizes
    if origin.size() != dest.size():
        raise ValueError("Size of origin (%s) does not match size of destination (%s)" % (origin.size(), dest.size()))
    # Contiguity
    if is_f_contig(H, strict=False):
        if not is_f_contig(D, strict=False):
            raise ValueError("origin is F-contig (strides %s), while destination is not (strides %s)" % (origin.stride(), dest.stride()))
    elif is_contig(H):  # H is C-contiguous
        if not is_contig(D) or is_f_contig(D, strict=True):
            raise ValueError("origin is C-contig (strides %s), while destination is not (strides %s)" % (origin.stride(), dest.stride()))
    else:
        raise ValueError("origin is not memory-contiguous (strides %s)" % (origin.stride(),))


def copy(origin, dest, s=None, allow_dtype_change=False):
    check_copy(origin, dest, check_dtypes=not allow_dtype_change)

    if origin.device.type == dest.device.type:
        dest.copy_(origin)
    elif origin.device.type == "cpu":  # host -> dev
        copy_to_device(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    else:                              # dev -> host
        copy_to_host(origin.shape[0], origin.shape[1], origin, 0, 0, dest, 0, 0, s)
    return dest


def copy_to_host(rows, cols, D, Di, Dj, H, Hi, Hj, s=None):
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)

    H_narrow_final = None
    if H.dtype != D.dtype:
        # First copy `D_narrow` to a matrix with the same dtype on the host
        # then copy this last matrix to `H_narrow` (at the end of this function).
        H_temp = torch.empty_like(D_narrow, device=D.device)
        H_narrow_final = H_narrow
        H_narrow = H_temp

    dts = sizeof_dtype(D.dtype)

    if is_f_contig(D, strict=False):
        if s is not None:
            cublasGetMatrixAsync(
                rows=rows, cols=cols, elem_size=dts,
                A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
                B=H_narrow.data_ptr(), ldb=H_narrow.stride(1),
                stream=s._as_parameter_)
        else:
            cublasGetMatrix(
                rows=rows, cols=cols, elem_size=dts,
                A=D_narrow.data_ptr(), lda=D_narrow.stride(1),
                B=H_narrow.data_ptr(), ldb=H_narrow.stride(1))
    elif is_contig(D):
        if s is not None:
            cuda_memcpy2d_async(
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter)
        else:
            cuda_memcpy2d(
                dst=H_narrow.data_ptr(), dpitch=H_narrow.stride(0) * dts,
                src=D_narrow.data_ptr(), spitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows)

    if H.dtype != D.dtype:
        H_narrow_final.copy_(H_narrow)  # Blocking copy since it's H->H.


def copy_to_device(rows, cols, H, Hi, Hj, D, Di, Dj, s=None):
    H_narrow = H.narrow(0, Hi, rows).narrow(1, Hj, cols)
    D_narrow = D.narrow(0, Di, rows).narrow(1, Dj, cols)

    if H.dtype != D.dtype:
        # First copy `H_narrow` to another matrix with correct dtype also on host
        # then copy this last matrix to `D`.
        H_right_dt = torch.empty_like(H_narrow, dtype=D.dtype)
        H_right_dt.copy_(H)  # Copy here will be blocking since it's H->H
        H_narrow = H_right_dt

    dts = sizeof_dtype(D.dtype)
    # strict needs to be False since cublas deals with row/column matrices just fine,
    # while cuda errors-out in certain cases (width > dpitch or width > spitch...)
    if is_f_contig(H, strict=False):
        if s is not None:
            cublasSetMatrixAsync(
                rows=rows, cols=cols, elem_size=dts,
                A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
                B=D_narrow.data_ptr(), ldb=D_narrow.stride(1),
                stream=s._as_parameter_)
        else:
            cublasSetMatrix(
                rows=rows, cols=cols, elem_size=dts,
                A=H_narrow.data_ptr(), lda=H_narrow.stride(1),
                B=D_narrow.data_ptr(), ldb=D_narrow.stride(1))
    elif is_contig(H):
        if s is not None:
            cuda_memcpy2d_async(
                src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
                dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows, stream=s._as_parameter)
        else:
            cuda_memcpy2d(
                src=H_narrow.data_ptr(), spitch=H_narrow.stride(0) * dts,
                dst=D_narrow.data_ptr(), dpitch=D_narrow.stride(0) * dts,
                width=cols * dts, height=rows)
    return D_narrow
