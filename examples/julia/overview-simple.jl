import Pkg
Pkg.add("PyCall")

using PyCall
using Printf

proxsuite = pyimport("proxsuite")
np = pyimport("numpy")
spa = pyimport("scipy.sparse")
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

EPS = 1e-6
# generate mixed random convex qp with size n
function generate_mixed_qp(n, seed=1)
    np.random.seed(seed)
    n_eq = trunc(Int, n / 4)
    n_in = trunc(Int, n / 4)
    m = n_eq + n_in

    P = spa.random(
        n, n, density=0.85, data_rvs=np.random.randn, format="csc"
        ).toarray()
    P = (P + np.transpose(P)) / 2.0

    s = np.max(np.absolute(np.linalg.eigvals(P)))
    P += (np.abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.85, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A * v
    l = -1.0e20 * np.ones(m)

    return P.toarray(), q, get(A, slice(0, n_eq)), get(u, slice(0, n_eq)), get(A, slice(n_in, m)), get(u, slice(n_in, m, )), l[n_in+1:end]
end

# generate a qp problem
n = 10
H, g, A, b, C, u, l = generate_mixed_qp(n)
n_eq = A.shape[1]
n_in = C.shape[1]

# solve it
qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
qp.settings.eps_abs = EPS
qp.init(H, g, A.toarray(), b, C.toarray(), l, u)
qp.solve()

x_res = qp.results.x
y_res = qp.results.y
z_res = qp.results.z

# calculate primal and dual residual
prim_res = max(
    np.linalg.norm(A * x_res - b, np.inf),
    np.linalg.norm(np.maximum(C * x_res - u, 0) + np.minimum(C * x_res - l, 0), np.inf)
)
dual_res = np.linalg.norm(H * x_res + g + np.transpose(A) * y_res + np.transpose(C) * z_res )

# assert that solved with required precision
@assert qp.results.info.pri_res < EPS
@assert qp.results.info.dua_res < EPS
@assert np.isclose(prim_res, qp.results.info.pri_res)
@assert np.isclose(dual_res, qp.results.info.dua_res)

# print stats
@printf("Done solving the qp with dim %i having %i equality and %i inequality constraints.\n", n, n_eq, n_in)
@printf("Primal residual: %f\n", qp.results.info.pri_res)
@printf("Dual residual: %f\n", qp.results.info.dua_res)
@printf("Total number of iteration %i.\n", qp.results.info.iter)
@printf("Setup time %fms, solve time %fms.\n", qp.results.info.setup_time, qp.results.info.solve_time)
@printf("Epsilon absolute %f, epsilon relative %f.\n", qp.settings.eps_abs, qp.settings.eps_rel)