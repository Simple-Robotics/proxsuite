#include "qp/dense/dense.hpp"
#include "test/include/util.hpp"
 
using namespace qp;
using T = double;
int main()
{
    // generate a QP problem
	T sparsity_factor = 0.15;
	dense::isize dim = 10;
	dense::isize n_eq(dim / 4);
	dense::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

    // load PROXQP solver with dense backend and solve the problem
	dense::QP<T> Qp(dim, n_eq, n_in); 
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();
}
