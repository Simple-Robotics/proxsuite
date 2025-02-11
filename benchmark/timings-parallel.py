import proxsuite
import numpy as np
import scipy.sparse as spa
from time import perf_counter_ns
from concurrent.futures import ThreadPoolExecutor

"""
There are two interfaces to solve a QP problem with the dense backend. a) create a qp object by passing the problem data (matrices, vectors) to the qp.init method (this does memory allocation and the preconditioning) and then calling qp.solve or b) use the solve function directly taking the problem data as input (this does everything in one go).

Currently, only the qp.solve method (a) is parallelized (using openmp). Therefore the memory alloc + preconditioning is done in serial when building a batch of qps that is then passed to the `solve_in_parallel` function. The solve function (b) is not parallelized but can easily be parallelized in Python using ThreadPoolExecutor.

Here we do some timings to compare the two approaches. We generate a batch of QP problems and solve them in parallel using the `solve_in_parallel` function and compare the timings (need to add the timings for building the batch of qps + the parallel solving) with solving each problem in parallel using ThreadPoolExecutor for the solve function.
"""

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
    _delta = np.random.rand(m)  # To get inequality
    u = A @ v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, A[:n_eq, :], u[:n_eq], A[n_in:, :], l[n_in:], u[n_in:]


problem_specs = [
    # (n, n_eq, n_in),
    (50, 20, 20),
    (100, 40, 40),
    (200, 80, 80),
    (500, 200, 200),
    (1000, 200, 200),
]

num_qps = 128

for n, n_eq, n_in in problem_specs:
    print(f"\nProblem specs: {n=} {n_eq=} {n_in=}. Generating {num_qps} such problems.")
    problems = [generate_mixed_qp(n, n_eq, n_in, seed=j) for j in range(num_qps)]
    print(
        f"Generated problems. Solving {num_qps} problems with proxsuite.proxqp.omp_get_max_threads()={num_threads} threads."
    )

    timings = {}

    # create a vector of QP objects. This is not efficient because memory is allocated when creating the qp object + when it is appended to the vector which creates a copy of the object.
    qps_vector = proxsuite.proxqp.dense.VectorQP()
    tic = perf_counter_ns()
    print("\nSetting up vector of qps")
    for H, g, A, b, C, l, u in problems:
        qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
        qp.init(H, g, A, b, C, l, u)
        qp.settings.eps_abs = 1e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qps_vector.append(qp)
    timings["setup_vector_of_qps"] = (perf_counter_ns() - tic) * 1e-6

    # use BatchQP, which can initialize the qp objects in place and is more efficient
    qps_batch = proxsuite.proxqp.dense.BatchQP()
    tic = perf_counter_ns()
    print("Setting up batch of qps")
    for H, g, A, b, C, l, u in problems:
        qp = qps_batch.init_qp_in_place(n, n_eq, n_in)
        qp.init(H, g, A, b, C, l, u)
        qp.settings.eps_abs = 1e-9
        qp.settings.verbose = False
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
    timings["setup_batch_of_qps"] = (perf_counter_ns() - tic) * 1e-6

    print("Solving batch of qps using solve_in_parallel with default thread config")
    tic = perf_counter_ns()
    proxsuite.proxqp.dense.solve_in_parallel(qps=qps_batch)
    timings["solve_in_parallel_heuristics_threads"] = (perf_counter_ns() - tic) * 1e-6

    print("Solving vector of qps serially")
    tic = perf_counter_ns()
    for qp in qps_vector:
        qp.solve()
    timings["qp_solve_serial"] = (perf_counter_ns() - tic) * 1e-6

    print("Solving batch of qps using solve_in_parallel with various thread configs")
    for j in range(1, num_threads, 2):
        tic = perf_counter_ns()
        proxsuite.proxqp.dense.solve_in_parallel(qps=qps_batch, num_threads=j)
        timings[f"solve_in_parallel_{j}_threads"] = (perf_counter_ns() - tic) * 1e-6

    def solve_problem_with_dense_backend(
        problem,
    ):
        H, g, A, b, C, l, u = problem
        return proxsuite.proxqp.dense.solve_no_gil(
            H,
            g,
            A,
            b,
            C,
            l,
            u,
            initial_guess=proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS,
            eps_abs=1e-9,
        )

    # add final timings for the solve_in_parallel function considering setup time for batch of qps
    for k, v in list(timings.items()):
        if "solve_in_parallel" in k:
            k_init = k + "_and_setup_batch_of_qps"
            timings[k_init] = timings["setup_batch_of_qps"] + v

    print("Solving each problem serially with solve function.")
    # Note: here we just pass the problem data to the solve function. This does not require running the init method separately.
    tic = perf_counter_ns()
    for problem in problems:
        solve_problem_with_dense_backend(problem)
    timings["solve_fun_serial"] = (perf_counter_ns() - tic) * 1e-6

    print(
        "Solving each problem in parallel (with a ThreadPoolExecutor) with solve function."
    )
    tic = perf_counter_ns()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(solve_problem_with_dense_backend, problems))
    timings["solve_fun_parallel"] = (perf_counter_ns() - tic) * 1e-6

    print("\nTimings:")
    for k, v in timings.items():
        print(f"{k}: {v:.3f}ms")
