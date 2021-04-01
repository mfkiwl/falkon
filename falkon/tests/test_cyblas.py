import time
from functools import partial

import numpy as np
import pytest
import scipy
import torch

from falkon.la_helpers import copy_triang, potrf, mul_triang, vec_mul_triang, zero_triang, trsm
from falkon.tests.conftest import fix_mat
from falkon.tests.gen_random import gen_random, gen_random_pd
from falkon.utils import decide_cuda
from falkon.utils.tensor_helpers import create_same_stride, move_tensor


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestCudaTranspose:
    t = 600

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random(self.t, self.t, np.float64, F=True, seed=12345)

    @pytest.fixture(scope="class")
    def rect(self):
        return gen_random(self.t, self.t * 2 - 1, np.float64, F=True, seed=12345)

    def test_square(self, mat, order, dtype):
        from falkon.la_helpers.cuda_la_helpers import cuda_transpose
        mat = fix_mat(mat, order=order, dtype=dtype, copy=True, numpy=True)
        mat_out = np.copy(mat, order="K")
        exp_mat_out = np.copy(mat.T, order=order)

        mat = move_tensor(torch.from_numpy(mat), "cuda:0")
        mat_out = move_tensor(torch.from_numpy(mat_out), "cuda:0")

        cuda_transpose(input=mat, output=mat_out)

        mat_out = move_tensor(mat_out, "cpu").numpy()
        assert mat_out.strides == exp_mat_out.strides
        np.testing.assert_allclose(exp_mat_out, mat_out)

    def test_rect(self, rect, order, dtype):
        from falkon.la_helpers.cuda_la_helpers import cuda_transpose
        mat = fix_mat(rect, order=order, dtype=dtype, copy=True, numpy=True)
        exp_mat_out = np.copy(mat.T, order=order)

        mat = move_tensor(torch.from_numpy(mat), "cuda:0")
        mat_out = move_tensor(torch.from_numpy(exp_mat_out), "cuda:0")
        mat_out.fill_(0.0)

        cuda_transpose(input=mat, output=mat_out)

        mat_out = move_tensor(mat_out, "cpu").numpy()
        assert mat_out.strides == exp_mat_out.strides
        np.testing.assert_allclose(exp_mat_out, mat_out)


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))])
class TestCopyTriang:
    t = 5

    @pytest.fixture(scope="class")
    def mat(self):
        return np.random.random((self.t, self.t))

    def test_low(self, mat, order, dtype, device):
        mat = fix_mat(mat, order=order, dtype=dtype, numpy=True)
        mat_low = mat.copy(order="K")
        # Upper triangle of mat_low is 0
        mat_low[np.triu_indices(self.t, 1)] = 0

        # Create device matrix
        mat_low = torch.from_numpy(mat_low)
        mat_low_dev = move_tensor(mat_low, device)

        # Run copy
        copy_triang(mat_low_dev, upper=False)

        # Make checks on CPU
        mat_low = mat_low_dev.cpu().numpy()
        assert np.sum(mat_low == 0) == 0
        np.testing.assert_array_equal(np.tril(mat), np.tril(mat_low))
        np.testing.assert_array_equal(np.triu(mat_low), np.tril(mat_low).T)
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_low))

        # Reset and try with `upper=True`
        mat_low[np.triu_indices(self.t, 1)] = 0
        mat_low_dev.copy_(torch.from_numpy(mat_low))

        copy_triang(mat_low_dev, upper=True)  # Only the diagonal will be set

        mat_low = mat_low_dev.cpu().numpy()
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_low))

    def test_up(self, mat, order, dtype, device):
        mat = fix_mat(mat, order=order, dtype=dtype, numpy=True)
        mat_up = mat.copy(order="K")
        # Lower triangle of mat_up is 0
        mat_up[np.tril_indices(self.t, -1)] = 0
        # Create device matrix
        mat_up = torch.from_numpy(mat_up)
        mat_up_dev = move_tensor(mat_up, device)

        copy_triang(mat_up_dev, upper=True)
        mat_up = mat_up_dev.cpu().numpy()

        assert np.sum(mat_up == 0) == 0
        np.testing.assert_array_equal(np.triu(mat), np.triu(mat_up))
        np.testing.assert_array_equal(np.tril(mat_up), np.triu(mat_up).T)
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_up))

        # Reset and try with `upper=False`
        mat_up[np.tril_indices(self.t, -1)] = 0
        mat_up_dev.copy_(torch.from_numpy(mat_up))

        copy_triang(mat_up_dev, upper=False)  # Only the diagonal will be set.

        mat_up = mat_up_dev.cpu().numpy()
        np.testing.assert_array_equal(np.diag(mat), np.diag(mat_up))


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "dirty"])
@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "copy"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestPotrf:
    t = 50
    rtol = {
        np.float64: 1e-13,
        np.float32: 1e-6
    }

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random_pd(self.t, np.float64, F=True, seed=12345)

    @pytest.fixture(scope="class")
    def exp_lower(self, mat):
        return np.linalg.cholesky(mat)

    @pytest.fixture(scope="class")
    def exp_upper(self, exp_lower):
        return exp_lower.T

    def test_upper(self, mat, exp_upper, clean, overwrite, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, copy=False, numpy=True)
        inpt = mat.copy(order="K")

        our_chol = potrf(inpt, upper=True, clean=clean, overwrite=overwrite, cuda=False)
        if overwrite:
            assert inpt.ctypes.data == our_chol.ctypes.data, "Overwriting failed"

        if clean:
            np.testing.assert_allclose(exp_upper, our_chol, rtol=self.rtol[dtype])
            assert np.tril(our_chol, -1).sum() == 0
        else:
            np.testing.assert_allclose(exp_upper, np.triu(our_chol), rtol=self.rtol[dtype])
            np.testing.assert_allclose(np.tril(mat, -1), np.tril(our_chol, -1))

    def test_lower(self, mat, exp_lower, clean, overwrite, order, dtype):
        mat = fix_mat(mat, order=order, dtype=dtype, copy=False, numpy=True)
        inpt = mat.copy(order="K")

        our_chol = potrf(inpt, upper=False, clean=clean, overwrite=overwrite, cuda=False)
        if overwrite:
            assert inpt.ctypes.data == our_chol.ctypes.data, "Overwriting failed"

        if clean:
            np.testing.assert_allclose(exp_lower, our_chol, rtol=self.rtol[dtype])
            assert np.triu(our_chol, 1).sum() == 0
        else:
            np.testing.assert_allclose(exp_lower, np.tril(our_chol), rtol=self.rtol[dtype])
            np.testing.assert_allclose(np.triu(mat, 1), np.triu(our_chol, 1))


@pytest.mark.benchmark
def test_potrf_speed():
    t = 5000
    mat = gen_random_pd(t, np.float32, F=False, seed=12345)
    t_s = time.time()
    our_chol = potrf(mat, upper=False, clean=True, overwrite=False, cuda=False)
    our_time = time.time() - t_s

    t_s = time.time()
    np_chol = np.linalg.cholesky(mat)
    np_time = time.time() - t_s

    np.testing.assert_allclose(np_chol, our_chol, rtol=1e-5)
    print("Time for cholesky(%d): Numpy %.2fs - Our %.2fs" % (t, np_time, our_time))


@pytest.mark.parametrize("preserve_diag", [True, False], ids=["preserve", "no-preserve"])
@pytest.mark.parametrize("upper", [True, False], ids=["upper", "lower"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))])
class TestMulTriang:
    t = 5

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random(self.t, self.t, np.float32, F=False, seed=123)

    def test_zero(self, mat, upper, preserve_diag, order, device):
        inpt1 = fix_mat(mat, dtype=mat.dtype, order=order, copy=True, numpy=True)
        inpt2 = inpt1.copy(order="K")

        k = 1 if preserve_diag else 0
        if upper:
            tri_fn = partial(np.triu, k=k)
        else:
            tri_fn = partial(np.tril, k=-k)

        inpt1 = torch.from_numpy(inpt1)
        inpt1_dev = create_same_stride(inpt1.shape, inpt1, inpt1.dtype, device)
        inpt1_dev.copy_(inpt1)
        mul_triang(inpt1_dev, upper=upper, preserve_diag=preserve_diag, multiplier=0)
        inpt1 = inpt1_dev.cpu().numpy()

        assert np.sum(tri_fn(inpt1)) == 0

        if preserve_diag:
            inpt2_dev = inpt1_dev
            inpt2_dev.copy_(torch.from_numpy(inpt2))
            zero_triang(inpt2_dev, upper=upper)
            inpt2 = inpt2_dev.cpu().numpy()
            np.testing.assert_allclose(inpt1, inpt2)

    def test_mul(self, mat, upper, preserve_diag, order, device):
        inpt1 = fix_mat(mat, dtype=mat.dtype, order=order, copy=True, numpy=True)

        k = 1 if preserve_diag else 0
        if upper:
            tri_fn = partial(np.triu, k=k)
            other_tri_fn = partial(np.tril, k=k - 1)
        else:
            tri_fn = partial(np.tril, k=-k)
            other_tri_fn = partial(np.triu, k=-k + 1)

        inpt1 = torch.from_numpy(inpt1)
        inpt1_dev = create_same_stride(inpt1.shape, inpt1, inpt1.dtype, device)
        inpt1_dev.copy_(inpt1)
        mul_triang(inpt1_dev, upper=upper, preserve_diag=preserve_diag, multiplier=10**6)
        inpt1 = inpt1_dev.cpu().numpy()

        assert np.mean(tri_fn(inpt1)) > 10**5
        assert np.mean(other_tri_fn(inpt1)) < 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order_v", ["C", "F"])
@pytest.mark.parametrize("order_A,device", [
    ("C", "cpu"), ("F", "cpu"),
    pytest.param("C", "cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found."),
                                       pytest.mark.xfail(reason="cuda TRSM expects F-contiguous A")]),
    pytest.param("F", "cuda:0", marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")])])
class TestTrsm:
    t = 50
    r = 10
    rtol = {
        np.float32: 1e-4,
        np.float64: 1e-12,
    }

    @pytest.fixture(scope="class")
    def alpha(self):
        return 1.0

    @pytest.fixture(scope="class")
    def mat(self):
        return gen_random(self.t, self.t, np.float64, F=True, seed=123)

    @pytest.fixture(scope="class")
    def vec(self):
        return gen_random(self.t, self.r, np.float64, F=True, seed=124)

    @pytest.fixture(scope="class", params=[
        (True, True), (True, False), (False, True), (False, False)], ids=[
        "lower-trans", "lower-no", "upper-trans", "upper-no"
    ])
    def solution(self, mat, vec, request):
        lower, trans = request.param
        return (scipy.linalg.solve_triangular(
            mat, vec, trans=int(trans), lower=lower, unit_diagonal=False,
            overwrite_b=False, debug=None, check_finite=True), lower, trans)

    def test_trsm(self, mat, vec, solution, alpha, dtype, order_v, order_A, device):
        mat = move_tensor(fix_mat(mat, dtype, order_A, copy=True, numpy=False), device=device)
        vec = move_tensor(fix_mat(vec, dtype, order_v, copy=True, numpy=False), device=device)

        sol_vec, lower, trans = solution
        out = trsm(vec, mat, alpha, lower=int(lower), transpose=int(trans))

        assert out.data_ptr() != vec.data_ptr(), "Vec was overwritten."
        assert out.device == vec.device, "Output device is incorrect."
        assert out.stride() == vec.stride(), "Stride was modified."
        assert out.dtype == vec.dtype, "Dtype was modified."
        np.testing.assert_allclose(sol_vec, out.cpu().numpy(), rtol=self.rtol[dtype])


class TestVecMulTriangSimple:
    @pytest.fixture
    def mat(self):
        arr = np.array([[1, 1, 1],
                        [2, 2, 4],
                        [6, 6, 8]], dtype=np.float32, order='F')
        return torch.from_numpy(arr).cuda()

    @pytest.fixture
    def vec(self):
        arr = np.array([0, 1, 0.5], dtype=np.float32)
        return torch.from_numpy(arr).cuda()

    def test(self, mat, vec):
        out = vec_mul_triang(mat, vec, True, 1)
        print("INPUT")
        print(mat.stride())
        print(mat)
        print("VEC")
        print(vec)
        print("EXPECTED")
        print(vec_mul_triang(mat.cpu(), vec.cpu(), True, 1))
        print("ACTUAL")
        print(out)



class TestVecMulTriang:
    @pytest.fixture
    def mat(self):
        return np.array([[1, 1, 1],
                         [2, 2, 4],
                         [6, 6, 8]], dtype=np.float32)

    @pytest.fixture
    def vec(self):
        return np.array([0, 1, 0.5], dtype=np.float32)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_lower(self, mat, vec, order):
        mat = fix_mat(mat, order=order, dtype=mat.dtype, numpy=True, copy=True)

        out = vec_mul_triang(mat.copy(order="K"), upper=False, side=0, multipliers=vec)
        exp = np.array([[0, 1, 1], [2, 2, 4], [3, 3, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

        out = vec_mul_triang(mat.copy(order="K"), upper=False, side=1, multipliers=vec)
        exp = np.array([[0, 1, 1], [0, 2, 4], [0, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_upper(self, mat, vec, order):
        mat = fix_mat(mat, order=order, dtype=mat.dtype, numpy=True, copy=True)

        out = vec_mul_triang(mat.copy(order="K"), upper=True, side=0, multipliers=vec)
        exp = np.array([[0, 0, 0], [2, 2, 4], [6, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

        out = vec_mul_triang(mat.copy(order="K"), upper=True, side=1, multipliers=vec)
        exp = np.array([[0, 1, 0.5], [2, 2, 2], [6, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(exp, out)
        assert out.flags["%s_CONTIGUOUS" % order] is True, "Output is not %s-contiguous" % (order)

    @pytest.mark.benchmark
    def test_large(self):
        t = 30_000
        mat = gen_random(t, t, np.float64, F=False, seed=123)
        vec = gen_random(t, 1, np.float64, F=False, seed=124).reshape((-1,))

        t_s = time.time()
        vec_mul_triang(mat, vec, upper=True, side=1)
        t_tri = time.time() - t_s

        t_s = time.time()
        mat *= vec
        t_full = time.time() - t_s

        print("Our took %.2fs -- Full took %.2fs" % (t_tri, t_full))
