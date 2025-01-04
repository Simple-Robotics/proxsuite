import proxsuite
import numpy as np
import scipy.sparse as spa
from time import perf_counter_ns
from concurrent.futures import ThreadPoolExecutor


num_threads = proxsuite.proxqp.omp_get_max_threads()


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

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], l[n_in:], u[n_in:]


n = 500
n_eq = 200
n_in = 200

num_qps = 128

print(f"Problem specs: {n=} {n_eq=} {n_in=}. Generating {num_qps} such problems.")
problems = [generate_mixed_qp(n, n_eq, n_in, seed=j) for j in range(num_qps)]
print(f"Generated problems. Solving {num_qps} problems with proxsuite.proxqp.omp_get_max_threads()={num_threads} threads.")

# qps = []
timings = {}
qps = proxsuite.proxqp.dense.VectorQP()

tic = perf_counter_ns()
print("Setting up problem vector")
for H, g, A, b, C, l, u in problems:
    qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
    qp.init(H, g, A, b, C, l, u)
    qp.settings.eps_abs = 1e-9
    qp.settings.verbose = False
    qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
    qps.append(qp)
timings["setup_vector_of_qps"] = (perf_counter_ns() - tic) * 1e-6

print("Solving problem vector in parallel with default thread config")
tic = perf_counter_ns()
proxsuite.proxqp.dense.solve_in_parallel(qps=qps)
timings[f"solve_parallel_heuristics_threads"] = (perf_counter_ns() - tic) * 1e-6

print("Solving problem vector serially")
tic = perf_counter_ns()
for qp in qps:
    qp.solve()
timings["solve_serial"] = (perf_counter_ns() - tic) * 1e-6

print("Solving problem vector in parallel with various thread configs")
for j in range(1, num_threads):
    tic = perf_counter_ns()
    proxsuite.proxqp.dense.solve_in_parallel(qps=qps, num_threads=j)
    timings[f"solve_parallel_{j}_threads"] = (perf_counter_ns() - tic) * 1e-6

print("Solving each problem serially with dense backend.")
tic = perf_counter_ns()
for H, g, A, b, C, l, u in problems:
    proxsuite.proxqp.dense.solve(H, g, A, b, C, l, u, initial_guess=proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS, eps_abs=1e-9)
timings["solve_serial_dense"] = (perf_counter_ns() - tic) * 1e-6

print("Solving each problem in parallel (with a ThreadPoolExecutor) with dense backend.")
def solve_problem(problem):  # just a little helper function to keep things clean
    H, g, A, b, C, l, u = problem
    return proxsuite.proxqp.dense.solve(H, g, A, b, C, l, u, initial_guess=proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS, eps_abs=1e-9)

tic = perf_counter_ns()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(executor.map(solve_problem, problems))
timings["solve_parallel_dense"] = (perf_counter_ns() - tic) * 1e-6

for k, v in timings.items():
    print(f"{k}: {v:.3f}ms")
