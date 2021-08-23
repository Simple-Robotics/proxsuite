#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as spa
import os
from sys import argv
from numpy.linalg import svd, eigh, qr, norm
from pathlib import Path
from functools import lru_cache

### generating random equality problems


def generate_equality_qp_pbl(n, seed):
    np.random.seed(seed)

    m = int(n / 2)

    # Generate problem data in OSQP format

    H_ = spa.random(n, n, density=0.15, data_rvs=np.random.randn, format="csc")
    H_ = H_.dot(H_.T).tocsc() + 1e-02 * spa.eye(n)
    g_ = np.random.randn(n)
    A_ = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    x_sol = np.random.randn(n)  # Create fictitious solution

    l_ = A_ @ x_sol
    u_ = np.copy(l_)

    n_eq = m

    QP_osqp = (H_, g_, A_, u_, l_, n_eq)

    # Transofrm data in Prox QP format

    n_in = 0
    C_ = np.zeros((n_in, n))
    d_ = np.zeros(n_in)
    b_ = l_.copy()

    qp_unscaled = (H_.toarray(), g_, A_.toarray(), b_, C_, d_)

    return qp_unscaled


@lru_cache(maxsize=1024)
def randomOrthonormal(dim, seed):
    np.random.seed(seed)
    mat = np.random.normal(0, 1, (dim, dim))
    q, r = qr(mat)
    return q


def randomDefinitePositive(dim, delta_l, seed):
    ### randomDefinitePositive: compute a Random Positive Definite Matrix with
    U = randomOrthonormal(dim, seed)

    s = np.power(10.0, delta_l * np.linspace(0, 1, dim))
    A = U * np.diag(s) * U.T
    return A


def generate_equality_qp_pbl_ill_conditionned(n, seed, delta_l=12):

    m = int(n / 2)

    # Generate problem data in OSQP format

    H_ = randomDefinitePositive(n, delta_l, seed)
    g_ = np.random.randn(n)
    A_ = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    x_sol = np.random.randn(n)  # Create fictitious solution

    l_ = A_ @ x_sol
    u_ = np.copy(l_)

    n_eq = m

    QP_osqp = (H_, g_, A_, u_, l_, n_eq)

    # Transofrm data in Prox QP format

    n_in = 0
    C_ = np.zeros((n_in, n))
    d_ = np.zeros(n_in)
    b_ = l_.copy()

    qp_unscaled = (H_, g_, A_.toarray(), b_, C_, d_)

    return qp_unscaled


### generate space parameter


def gen_int_log_space(min_val, limit, n):
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem same integer. we need to find next_value
            # by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale
            # correctly
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1 + min_val, result)), dtype=int)


def stock_problem(dim_l, seed_l, path, name, ill_conditionned, delta_l):

    l_dir_path = []
    for seed in seed_l:
        list_keys = []
        np.random.seed(seed)
        fname = name + "_seed_" + str(seed)
        if ill_conditionned:
            fname += "_delta_l_" + str(delta_l)

        for dim in dim_l:

            if not ill_conditionned:
                qp_prox = generate_equality_qp_pbl(dim, seed)
            else:
                qp_prox = generate_equality_qp_pbl_ill_conditionned(dim, delta_l, seed)

            current_qp_path = path / (
                fname
                + "_dim_"
                + str(qp_prox[0].shape[0])
                + "_n_eq_"
                + str(qp_prox[2].shape[0])
            )
            current_qp_path.mkdir(exist_ok=True)
            list_keys.append(current_qp_path)

            name_H = current_qp_path / "H"
            np.save(name_H, qp_prox[0])
            name_g = current_qp_path / "g"
            np.save(name_g, qp_prox[1])
            name_A = current_qp_path / "A"
            np.save(name_A, qp_prox[2])
            name_b = current_qp_path / "b"
            np.save(name_b, qp_prox[3])

        l_dir_path += list_keys

    return l_dir_path


def generate_problems(dim_l, l_seed, path, l_pbl_type, l_delta_l):
    list_dir_path = []
    current_path = path / "qp_problem"
    current_path.mkdir(parents=True, exist_ok=True)

    for pbl_type in l_pbl_type:
        if pbl_type:
            name = "ill_conditionned"

            for delta_l in l_delta_l:
                l_path = stock_problem(
                    dim_l, l_seed, current_path, name, pbl_type, delta_l
                )
                list_dir_path += l_path
        else:
            name = "osqp_pbl"
            for delta_l in l_delta_l:
                l_path = stock_problem(
                    dim_l, l_seed, current_path, name, pbl_type, delta_l
                )
                list_dir_path += l_path
    with open(current_path / "source_files.txt", "w") as f:
        f.write("\n".join([p.absolute().as_posix() for p in list_dir_path]))


if __name__ == "__main__":
    ### problem dimensions

    n_dim = 20
    dim_l = gen_int_log_space(10, 2000, n_dim)
    l_min = 0
    l_max = 12

    ######## Random Problems ##########

    ## generating files

    path = Path(argv[1])

    l_pbl_type = [
        True,
        False,
    ]  # when True generates ill_conditionned problem, when False generates osqp type problems
    l_delta_l = [12]  # maximum power value of conditionned number
    l_seed = [0, 1]

    generate_problems(dim_l, l_seed, path, l_pbl_type, l_delta_l)
