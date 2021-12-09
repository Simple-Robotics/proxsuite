#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "qp/views.hpp"
#include <qp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>
#include <util.hpp>

using namespace ldlt;

using Scalar = long double;

DOCTEST_TEST_CASE("qp: random") {
	isize dim = 30;
	isize n_eq = 6;
    isize n_in = 6;
	isize max_it = 200;
	isize max_it_in = 2500;

	QpBox<Scalar> qp{random_with_dim_and_n_eq_and_n_in, dim, n_eq,n_in};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init_eq(n_eq);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init_in(n_in);
	primal_init.setZero();
	primal_init = -qp.H.llt().solve(qp.g);
	dual_init_eq.setZero();
    dual_init_in.setZero();

	Scalar eps_abs = Scalar(1e-9);
	Scalar eps_rel = Scalar(0);
	{
		EigenNoAlloc _{};
		qp::detail::solve_qp_in( //
				VectorViewMut<Scalar>{from_eigen, primal_init},
				VectorViewMut<Scalar>{from_eigen, dual_init_eq},
                VectorViewMut<Scalar>{from_eigen, dual_init_in},
				qp.as_view_box(),
				max_it,
				max_it_in,
				eps_abs,
				eps_rel,
				qp::preconditioner::IdentityPrecond{});
	}

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init_eq + qp.C.transpose() * dual_init_in)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}