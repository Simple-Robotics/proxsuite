using PyCall

proxsuite = pyimport("proxsuite")
np = pyimport("numpy")
spa = pyimport("scipy.sparse")
slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)


# generate mixed random convex qp with size n
function generate_mixed_qp(n, seed=1)
    np.random.seed(seed)
    n_eq = trunc(Int, n / 4)
    n_in = trunc(Int, n / 4)
    m = n_eq + n_in

    P = spa.random(
        n, n, density=0.075, data_rvs=np.random.randn, format="csc"
        ).toarray()
    P = (P + np.transpose(P)) / 2.0
    
    s = np.max(np.absolute(np.linalg.eigvals(P)))
    P += (np.abs(s) + 1e-02) * spa.eye(n)
    P = spa.coo_matrix(P)
    q = np.random.randn(n)
    A = spa.random(m, n, density=0.15, data_rvs=np.random.randn, format="csc")
    v = np.random.randn(n)  # Fictitious solution
    delta = np.random.rand(m)  # To get inequality
    u = A * v
    l = -1.0e20 * np.ones(m)

    return P, q, get(A, slice(0, n_eq)), get(u, slice(0, n_eq)), get(A, slice(n_in, -1)), get(u, slice(n_in, -1, )), l[n_in+1:end]
end

# generate a qp problem
n = 10
H, g, A, b, C, u, l = generate_mixed_qp(n)
n_eq = A.shape[1]
n_in = C.shape[1]

# solve it
Qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)
Qp.init(H, g, A, b, C, u, l)
Qp.solve()
