import dataclasses

import numpy as np
import pytest
import torch

from falkon.kernels import *
from falkon.options import FalkonOptions
from falkon.tests.conftest import memory_checker, fix_mats
from falkon.tests.gen_random import gen_random, gen_sparse_matrix
from falkon.tests.naive_kernels import *
from falkon.utils import decide_cuda
from falkon.utils.switches import decide_keops
from falkon.utils.helpers import sizeof_dtype

cuda_mark = pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
keops_mark = pytest.mark.skipif(not decide_keops(), reason="no KeOps found.")
device_marks = [
    pytest.param("cpu", "cpu"),
    pytest.param("cpu", "cuda", marks=[cuda_mark]),
    pytest.param("cuda", "cuda", marks=[cuda_mark])
]
# Global data dimensions
n = 20
m = 5
d = 3
t = 2
# Sparse data dimensions
s_n = 500
s_m = 550
s_d = 20000
density = 1e-5

max_mem = 2 * 2 ** 20
basic_options = FalkonOptions(debug=True, compute_arch_speed=False,
                              max_cpu_mem=max_mem, max_gpu_mem=max_mem)


@pytest.fixture(scope="module")
def A() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, d, 'float32', False, seed=92))


@pytest.fixture(scope="module")
def B() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, d, 'float32', False, seed=93))


@pytest.fixture(scope="module")
def v() -> torch.Tensor:
    return torch.from_numpy(gen_random(m, t, 'float32', False, seed=94))


@pytest.fixture(scope="module")
def w() -> torch.Tensor:
    return torch.from_numpy(gen_random(n, t, 'float32', False, seed=95))


@pytest.fixture(scope="module")
def s_A():
    A = gen_sparse_matrix(s_n, s_d, np.float64, density=density, seed=14)
    Ad = torch.from_numpy(A.to_scipy().todense())
    return A, Ad


@pytest.fixture(scope="module")
def s_B():
    B = gen_sparse_matrix(s_m, s_d, np.float64, density=density, seed=14)
    Bd = torch.from_numpy(B.to_scipy().todense())
    return B, Bd


@pytest.fixture(scope="module")
def s_v() -> torch.Tensor:
    return torch.from_numpy(gen_random(s_m, t, 'float32', False, seed=94))


@pytest.fixture(scope="module")
def s_w() -> torch.Tensor:
    return torch.from_numpy(gen_random(s_n, t, 'float32', False, seed=95))


@pytest.fixture(scope="module")
def rtol():
    return {
        np.float64: 1e-12,
        torch.float64: 1e-12,
        np.float32: 1e-4,
        torch.float32: 1e-4
    }


@pytest.fixture(scope="module")
def atol():
    return {
        np.float64: 1e-12,
        torch.float64: 1e-12,
        np.float32: 1e-4,
        torch.float32: 1e-4
    }


@pytest.fixture(params=["single-sigma", "vec-sigma"], scope="class")
def sigma(request) -> torch.Tensor:
    if request.param == "single-sigma":
        return torch.Tensor([3.0])
    elif request.param == "vec-sigma":
        return torch.Tensor([3.0] * d)


def run_sparse_test(k_cls, naive_fn, s_m1, s_m2, m1, m2, v, w, rtol, atol, opt, **kernel_params):
    kernel = k_cls(**kernel_params)

    # 1. MM
    mm_out = torch.empty(s_m1.shape[0], s_m2.shape[0], dtype=s_m1.dtype, device=s_m1.device)
    with memory_checker(opt) as new_opt:
        actual = kernel(s_m1, s_m2, out=mm_out, opt=new_opt)
    with memory_checker(opt,
                        extra_mem=m1.shape[0] * m2.shape[0] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel(s_m1, s_m2, opt=new_opt)
    assert mm_out.data_ptr() == actual.data_ptr(), "sparse MM Output data tensor was not used"
    torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                  msg="sparse MM with out and without return different stuff")
    expected_mm = naive_fn(m1, m2, **kernel_params)
    torch.testing.assert_allclose(expected_mm, actual, rtol=rtol, atol=atol,
                                  msg="sparse MM result is incorrect")

    # 2. MMV
    mmv_out = torch.empty(s_m1.shape[0], v.shape[1], dtype=s_m1.dtype, device=s_m1.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.mmv(s_m1, s_m2, v, out=mmv_out, opt=new_opt)
    with memory_checker(opt,
                        extra_mem=m1.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.mmv(s_m1, s_m2, v, opt=new_opt)
    assert mmv_out.data_ptr() == actual.data_ptr(), "sparse MMV Output data tensor was not used"
    torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                  msg="sparse MMV with out and without return different stuff")
    expected_mmv = expected_mm @ v
    torch.testing.assert_allclose(expected_mmv, actual, rtol=rtol, atol=atol,
                                  msg="sparse MMV result is incorrect")

    # 3. dMMV
    dmmv_out = torch.empty(s_m2.shape[0], v.shape[1], dtype=s_m2.dtype, device=s_m2.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.dmmv(s_m1, s_m2, v, w, out=dmmv_out, opt=new_opt)
    with memory_checker(opt,
                        extra_mem=m2.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.dmmv(s_m1, s_m2, v, w, opt=new_opt)
    assert dmmv_out.data_ptr() == actual.data_ptr(), "sparse D-MMV Output data tensor was not used"
    torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                  msg="sparse D-MMV with out and without return different stuff")
    expected_dmmv = expected_mm.T @ (expected_mmv + w)
    torch.testing.assert_allclose(expected_dmmv, actual, rtol=rtol, atol=atol,
                                  msg="sparse D-MMV result is incorrect")


def run_dense_test(k_cls, naive_fn, m1, m2, v, w, rtol, atol, opt,
                   grad_check: bool = True, **kernel_params):
    torch.autograd.set_detect_anomaly(True)

    kernel = k_cls(**kernel_params)

    m1_wgrad = m1.clone().requires_grad_()
    m2_wgrad = m2.clone().requires_grad_()
    v_wgrad = v.clone().requires_grad_()
    w_wgrad = w.clone().requires_grad_()
    kernel_params_wgrad = {k: v.clone().requires_grad_() for k, v in kernel.diff_params.items()}

    kernel_wgrad = k_cls(**kernel.nondiff_params, **kernel_params_wgrad)

    # FIXME: On some systems (nest but not sperone), checking memory
    #        usage for CPU functions fails miserably due to inconsistent
    #        memory numbers being reported at random. We simply replace CPU
    #        with a high number to avoid checking.
    extra_mem = 10 * 2 ** 30 if opt.use_cpu else 0
    opt = dataclasses.replace(opt, max_cpu_mem=opt.max_cpu_mem + extra_mem)

    expected_mm = naive_fn(m1, m2, **kernel_params)
    # 1. MM
    if opt.keops_active != "force":  # Don't test MM if keops is active
        mm_out = torch.empty(m1.shape[0], m2.shape[0], dtype=m1.dtype, device=m1.device)
        mm_out_wgrad = torch.empty(m1.shape[0], m2.shape[0], dtype=m1.dtype, device=m1.device)
        with memory_checker(opt) as new_opt:
            actual = kernel(m1, m2, out=mm_out, opt=new_opt)
        with memory_checker(opt, extra_mem=m1.shape[0] * m2.shape[0] * sizeof_dtype(
                m1.dtype)) as new_opt:
            actual_noout = kernel(m1, m2, opt=new_opt)
        with memory_checker(opt) as new_opt:
            actual_wgrad = kernel_wgrad(m1_wgrad, m2_wgrad, out=mm_out_wgrad, opt=new_opt)

        assert mm_out.data_ptr() == actual.data_ptr(), "MM Output data tensor was not used"
        assert mm_out_wgrad.data_ptr() == actual_wgrad.data_ptr(), "MM Output data tensor was not used"
        torch.testing.assert_allclose(actual_wgrad, actual, rtol=rtol, atol=atol,
                                      msg="MM Wgrad and normal return different stuff")
        torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                      msg="MM with out and without return different stuff")
        torch.testing.assert_allclose(expected_mm, actual, rtol=rtol, atol=atol,
                                      msg="MM result is incorrect")

        # 2. MM gradients
        if grad_check:
            def autogradcheck_mm(_m1, _m2, *_kernel_params):
                return k_cls(*_kernel_params, **kernel.nondiff_params, opt=opt)(_m1, _m2)

            torch.autograd.gradcheck(autogradcheck_mm,
                                     inputs=(m1_wgrad, m2_wgrad, *kernel_params_wgrad.values()))

    # 3. MMV
    mmv_out = torch.empty(m1.shape[0], v.shape[1], dtype=m1.dtype, device=m1.device)
    mmv_out_wgrad = torch.empty(m1.shape[0], v.shape[1], dtype=m1.dtype, device=m1.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.mmv(m1, m2, v, out=mmv_out, opt=new_opt)
    with memory_checker(opt,
                        extra_mem=m1.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.mmv(m1, m2, v, opt=new_opt)
    with memory_checker(opt) as new_opt:
        actual_wgrad = kernel_wgrad.mmv(m1_wgrad, m2_wgrad, v_wgrad, out=mmv_out_wgrad, opt=new_opt)
    assert mmv_out.data_ptr() == actual.data_ptr(), "MMV Output data tensor was not used"
    assert mmv_out_wgrad.data_ptr() == actual_wgrad.data_ptr(), "MMV Output data tensor was not used"
    torch.testing.assert_allclose(actual_wgrad, actual, rtol=rtol, atol=atol,
                                  msg="MMV Wgrad and normal return different stuff")
    torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                  msg="MMV with out and without return different stuff")
    expected_mmv = expected_mm @ v
    torch.testing.assert_allclose(expected_mmv, actual, rtol=rtol, atol=atol,
                                  msg="MMV result is incorrect")

    # 4. MMV gradients
    if grad_check:
        def autogradcheck_mmv(_m1, _m2, _v, *_kernel_params):
            return k_cls(*_kernel_params, **kernel.nondiff_params, opt=opt).mmv(_m1, _m2, _v)

        torch.autograd.gradcheck(autogradcheck_mmv, inputs=(
        m1_wgrad, m2_wgrad, v_wgrad, *kernel_params_wgrad.values()))

    # 5. Double MMV (doesn't exist for gradients)
    dmmv_grad_allowed = True
    dmmv_out = torch.empty(m2.shape[0], v.shape[1], dtype=m1.dtype, device=m1.device)
    with memory_checker(opt) as new_opt:
        actual = kernel.dmmv(m1, m2, v, w, out=dmmv_out, opt=new_opt)
    with memory_checker(opt,
                        extra_mem=m2.shape[0] * v.shape[1] * sizeof_dtype(m1.dtype)) as new_opt:
        actual_noout = kernel.dmmv(m1, m2, v, w, opt=new_opt)
    with memory_checker(opt) as new_opt:
        try:
            actual_wgrad = kernel_wgrad.dmmv(m1_wgrad, m2_wgrad, v_wgrad, w_wgrad, opt=new_opt)
        except NotImplementedError as e:
            assert new_opt.keops_active == "no", "KeOps D-MMV raise error %s unexpectedly" % (e)
            # On the other hand it is expected that we throw a not implemented error.
            dmmv_grad_allowed = False

    assert dmmv_out.data_ptr() == actual.data_ptr(), "D-MMV Output data tensor was not used"
    if dmmv_grad_allowed:
        torch.testing.assert_allclose(actual_wgrad, actual, rtol=rtol, atol=atol,
                                      msg="MMV Wgrad and normal return different stuff")
    torch.testing.assert_allclose(actual_noout, actual, rtol=rtol, atol=atol,
                                  msg="D-MMV with out and without return different stuff")
    expected_dmmv = expected_mm.T @ (expected_mmv + w)
    torch.testing.assert_allclose(expected_dmmv, actual, rtol=rtol, atol=atol,
                                  msg="D-MMV result is incorrect")

    # 6. D-MMV gradients
    if grad_check and dmmv_grad_allowed:
        def autogradcheck_dmmv(_m1, _m2, _v, _w, *_kernel_params):
            return k_cls(*_kernel_params, **kernel.nondiff_params, opt=opt).dmmv(_m1, _m2, _v, _w)

        torch.autograd.gradcheck(autogradcheck_dmmv, inputs=(
        m1_wgrad, m2_wgrad, v_wgrad, w_wgrad, *kernel_params_wgrad.values()))


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestLaplacianKernel():
    naive_fn = naive_diff_laplacian_kernel
    k_class = LaplacianKernel

    @pytest.fixture(scope="class")
    def rtol(self):
        return {
            torch.float32: 1e-5,
            torch.float64: 4e-8,
        }

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_dense_kernel(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev, order):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order=order, device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_dense_test(TestLaplacianKernel.k_class, TestLaplacianKernel.naive_fn, m1=A, m2=B,
                       v=v, w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma,
                       grad_check=True)

    def test_sparse_kernel(self, s_A, s_B, s_v, s_w, sigma, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, s_v, s_w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, s_v, s_w, sigma, order="C", device=input_dev, dtype=np.float32)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        try:
            run_sparse_test(TestLaplacianKernel.k_class, TestLaplacianKernel.naive_fn,
                            s_m1=s_A, s_m2=s_B, m1=A, m2=B, v=s_v, w=s_w, rtol=rtol[A.dtype],
                            atol=atol[A.dtype], opt=opt, sigma=sigma)
        except NotImplementedError:
            assert len(sigma) > 1, "NotImplementedError thrown with scalar sigma"
        else:
            assert len(sigma) == 1, "Sparse kernel worked unexpectedly with non-scalar sigma"

    def test_not_all_grads(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        m1, m2, v, w, sigma = fix_mats(A, B, v, w, sigma, order="F", device=input_dev,
                                       dtype=np.float64)

        m1_wgrad = m1.clone().requires_grad_(False)
        m2_wgrad = m2.clone().requires_grad_()
        v_wgrad = v.clone().requires_grad_(False)
        s_wgrad = sigma.clone().requires_grad_()

        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")

        def autogradcheck_mm(_m1, _m2, *_kernel_params):
            return TestLaplacianKernel.k_class(*_kernel_params, opt=opt)(_m1, _m2)

        torch.autograd.gradcheck(autogradcheck_mm, inputs=(m1_wgrad, m2_wgrad, s_wgrad))

        def autogradcheck_mmv(_m1, _m2, _v, *_kernel_params):
            return TestLaplacianKernel.k_class(*_kernel_params, opt=opt).mmv(_m1, _m2, _v)

        torch.autograd.gradcheck(autogradcheck_mmv, inputs=(m1_wgrad, m2_wgrad, v_wgrad, s_wgrad))

    @keops_mark
    def test_keops_kernel(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="C", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="force")
        run_dense_test(TestLaplacianKernel.k_class, TestLaplacianKernel.naive_fn, m1=A, m2=B,
                       v=v, w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma)

    @keops_mark
    def test_keops_kernel_noncontig(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="F", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="force")
        run_dense_test(TestLaplacianKernel.k_class, TestLaplacianKernel.naive_fn, m1=A, m2=B,
                       v=v, w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma)
        # TODO: Assert warning printed


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestGaussianKernel():
    naive_fn = naive_diff_gaussian_kernel
    k_class = GaussianKernel

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_dense_kernel(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev, order):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order=order, device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_dense_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn, m1=A, m2=B, v=v,
                       w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma)

    def test_sparse_kernel(self, s_A, s_B, s_v, s_w, sigma, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, s_v, s_w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, s_v, s_w, sigma, order="C", device=input_dev, dtype=np.float32)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        try:
            run_sparse_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn,
                            s_m1=s_A, s_m2=s_B, m1=A, m2=B, v=s_v, w=s_w, rtol=rtol[A.dtype],
                            atol=atol[A.dtype], opt=opt, sigma=sigma)
        except NotImplementedError:
            assert len(sigma) > 1, "NotImplementedError thrown with scalar sigma"
        else:
            assert len(sigma) == 1, "Sparse kernel worked unexpectedly with non-scalar sigma"

    @keops_mark
    def test_keops_kernel(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="F", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="force")
        run_dense_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn, m1=A, m2=B, v=v,
                       w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma)

    def test_wrong_sigma_dims(self, A, B, v, w, rtol, atol, input_dev, comp_dev):
        sigma = torch.tensor([2.0] * (d - 1), dtype=torch.float64)
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="F", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        with pytest.raises(RuntimeError) as excinfo:
            run_dense_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn, m1=A, m2=B, v=v,
                           w=w,
                           rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma)
        if comp_dev == "cpu":
            assert f"The size of tensor a ({d}) must match the size of tensor b ({d - 1})" in str(
                excinfo.value)
        # If on GPU the 'size mismatch' message is in the base exception (since it's reraised
        # by PropagatingThread) but I'm not sure how to fetch it.


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestMaternKernel():
    naive_fn = naive_diff_matern_kernel
    k_class = MaternKernel

    @pytest.fixture(params=[0.5, 1.5, 2.5, np.inf], scope="function")
    def nu(self, request) -> torch.Tensor:
        return torch.tensor(request.param)

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_dense_kernel(self, A, B, v, w, nu, sigma, rtol, atol, input_dev, comp_dev, order):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order=order, device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_dense_test(TestMaternKernel.k_class, TestMaternKernel.naive_fn, m1=A, m2=B, v=v, w=w,
                       rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma, nu=nu)

    def test_sparse_kernel(self, s_A, s_B, s_v, s_w, nu, sigma, rtol, atol, input_dev, comp_dev):
        s_A, d_A = s_A
        s_B, d_B = s_B
        s_A, A, s_B, B, s_v, s_w, sigma = fix_mats(
            s_A, d_A, s_B, d_B, s_v, s_w, sigma, order="C", device=input_dev, dtype=np.float32)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        try:
            run_sparse_test(TestMaternKernel.k_class, TestMaternKernel.naive_fn,
                            s_m1=s_A, s_m2=s_B, m1=A, m2=B, v=s_v, w=s_w, rtol=rtol[A.dtype],
                            atol=atol[A.dtype], opt=opt, sigma=sigma, nu=nu)
        except NotImplementedError:
            assert len(sigma) > 1, "NotImplementedError thrown with scalar sigma"
        else:
            assert len(sigma) == 1, "Sparse kernel worked unexpectedly with non-scalar sigma"

    @keops_mark
    def test_keops_kernel(self, A, B, v, w, nu, sigma, rtol, atol, input_dev, comp_dev):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="C", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="force")
        run_dense_test(TestMaternKernel.k_class, TestMaternKernel.naive_fn, m1=A, m2=B, v=v, w=w,
                       rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma, nu=nu)

    def test_nu_fail(self, A, B, v, w, rtol, atol, input_dev, comp_dev):
        sigma = torch.tensor([1.2])
        nu = torch.tensor(2.1)
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="F", device=input_dev,
                                     dtype=np.float64)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        with pytest.raises(ValueError) as excinfo:
            run_dense_test(TestMaternKernel.k_class, TestMaternKernel.naive_fn, m1=A, m2=B, v=v,
                           w=w,
                           rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma, nu=nu)
        assert f"The given value of nu = {nu:.1f} can only take values" in str(excinfo.value)


@pytest.mark.parametrize("input_dev,comp_dev", device_marks)
class TestLargeComputations():
    naive_fn = naive_diff_gaussian_kernel
    k_class = GaussianKernel
    n = 1500
    m = 250
    d = 3
    t = 2
    max_mem = 1 * 2 ** 20
    basic_options = FalkonOptions(debug=True, compute_arch_speed=False,
                                  max_cpu_mem=max_mem, max_gpu_mem=max_mem)
    sigma = torch.Tensor([3.0])

    @pytest.fixture(scope="class")
    def A(self) -> torch.Tensor:
        return torch.from_numpy(gen_random(self.n, self.d, 'float32', False, seed=92))

    @pytest.fixture(scope="class")
    def B(self) -> torch.Tensor:
        return torch.from_numpy(gen_random(self.m, self.d, 'float32', False, seed=93))

    @pytest.fixture(scope="class")
    def v(self) -> torch.Tensor:
        return torch.from_numpy(gen_random(self.m, self.t, 'float32', False, seed=94))

    @pytest.fixture(scope="class")
    def w(self) -> torch.Tensor:
        return torch.from_numpy(gen_random(self.n, self.t, 'float32', False, seed=95))

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_dense_kernel(self, A, B, v, w, rtol, atol, input_dev, comp_dev, order):
        A, B, v, w, sigma = fix_mats(A, B, v, w, self.sigma, order=order, device=input_dev,
                                     dtype=np.float32)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="no")
        run_dense_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn, m1=A, m2=B, v=v,
                       w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma,
                       grad_check=False)

    @keops_mark
    def test_keops_kernel(self, A, B, v, w, sigma, rtol, atol, input_dev, comp_dev):
        A, B, v, w, sigma = fix_mats(A, B, v, w, sigma, order="C", device=input_dev,
                                     dtype=np.float32)
        opt = dataclasses.replace(basic_options, use_cpu=comp_dev == "cpu", keops_active="force")
        run_dense_test(TestGaussianKernel.k_class, TestGaussianKernel.naive_fn, m1=A, m2=B, v=v,
                       w=w, rtol=rtol[A.dtype], atol=atol[A.dtype], opt=opt, sigma=sigma,
                       grad_check=False)


if __name__ == "__main__":
    pytest.main()
