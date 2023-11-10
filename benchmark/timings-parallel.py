import proxsuite
import numpy as np
import scipy.sparse as spa
from time import perf_counter_ns


def generate_mixed_qp(n, n_eq, n_in, seed=1):
    np.random.seed(seed)
    m = n_eq + n_in

    P = spa.random(
        n, n, density=0.075, data_rvs=np.random.randn, format="csc"
    ).toarray()
    P = (P + P.T) / 2.0

    s = max(np.absolute(np.linalg.eigvals(P)))
    P += (abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc").toarray()
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], u[n_in:], l[n_in:]


n = 500
n_eq = 200
n_in = 200

num_qps = 128

# qps = []
timings = {}
qps = proxsuite.proxqp.dense.VectorQP()

tic = perf_counter_ns()
for j in range(num_qps):
    qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    H, g, A, b, C, u, l = generate_mixed_qp(n, n_eq, n_in, seed=j)
    qp.init(H, g, A, b, C, l, u)
    qp.settings.eps_abs = 1e-9
    qp.settings.verbose = False
    qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
    qps.append(qp)
timings["problem_data"] = (perf_counter_ns() - tic) * 1e-6

tic = perf_counter_ns()
for qp in qps:
    qp.solve()
timings["solve_serial"] = (perf_counter_ns() - tic) * 1e-6

num_threads = proxsuite.proxqp.omp_get_max_threads()
for j in range(1, num_threads):
    tic = perf_counter_ns()
    proxsuite.proxqp.dense.solve_in_parallel(j, qps)
    timings[f"solve_parallel_{j}_threads"] = (perf_counter_ns() - tic) * 1e-6


tic = perf_counter_ns()
proxsuite.proxqp.dense.solve_in_parallel(qps=qps)
timings[f"solve_parallel_heuristics_threads"] = (perf_counter_ns() - tic) * 1e-6

for k, v in timings.items():
    print(f"{k}: {v}ms")
