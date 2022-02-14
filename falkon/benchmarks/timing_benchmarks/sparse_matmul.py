import time

import scipy.sparse
import torch
import numpy as np

from falkon.mkl_bindings.mkl_bind import mkl_lib
from falkon.tests.gen_random import random_sparse


def run(n, density, reps):
    sp_mat: scipy.sparse.csr_matrix = random_sparse(
        n, n, density=density, sparse_format='csr', dtype=np.float32, seed=132, data_rvs=None)
    dn_mat = np.asarray(sp_mat.todense())

    # Scipy
    scipy_times = []
    for i in range(reps):
        t_s = time.time()
        out = sp_mat.dot(sp_mat.transpose())
        t_e = time.time()
        scipy_times.append(t_e - t_s)
    print("Scipy times: %.4fs" % (np.min(scipy_times)))

    # Dense
    # dense_times = []
    # for i in range(reps):
    #     t_s = time.time()
    #     out = dn_mat @ dn_mat.T
    #     t_e = time.time()
    #     dense_times.append(t_e - t_s)
    # print("Dense times: %.4fs" % (np.min(dense_times)))

    # MKL
    mkl = mkl_lib()
    mkl_times = []
    for i in range(reps):
        t_s = time.time()
        sp_mat_t = sp_mat.transpose()
        out = torch.empty((n, n), dtype=torch.float32)
        mkl_sp_1 = mkl.mkl_create_sparse_from_scipy(sp_mat)
        mkl_sp_2 = mkl.mkl_create_sparse_from_scipy(sp_mat_t)
        mkl.mkl_spmmd(mkl_sp_1, mkl_sp_2, out, transposeA=True)
        t_e = time.time()
        mkl_times.append(t_e - t_s)
    print("MKL times: %.4fs" % (np.min(mkl_times)))


if __name__ == "__main__":
    run(100, 0.1, 5)
