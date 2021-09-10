#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/eq_solver.hpp>
#include <qp/precond/ruiz.hpp>
#include <util.hpp>

using namespace ldlt;

using Scalar = long double;

DOCTEST_TEST_CASE("qp: random") {
	isize dim = 30;
	isize n_eq = 6;

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init.setZero();
	primal_init = -qp.H.llt().solve(qp.g);
	dual_init.setZero();

	Scalar eps_abs = Scalar(1e-10);
	{
		EigenNoAlloc _{};
		qp::detail::solve_qp( //
				{from_eigen, primal_init},
				{from_eigen, dual_init},
				qp.as_view(),
				200,
				eps_abs,
				0,
				qp::preconditioner::IdentityPrecond{});
	}

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}

DOCTEST_TEST_CASE("qp: ruiz preconditioner") {
	isize dim = 30;
	isize n_eq = 6;

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init.setZero();
	primal_init = -qp.H.llt().solve(qp.g);
	dual_init.setZero();

	Scalar eps_abs = Scalar(1e-10);
	{
		auto ruiz = qp::preconditioner::RuizEquilibration<Scalar>{
				dim,
				n_eq,
		};
		EigenNoAlloc _{};
		qp::detail::solve_qp( //
				{from_eigen, primal_init},
				{from_eigen, dual_init},
				qp.as_view(),
				200,
				eps_abs,
				0,
				LDLT_FWD(ruiz));
	}

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}

DOCTEST_TEST_CASE("qp: start from solution") {
	isize dim = 30;
	isize n_eq = 6;

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init = qp.solution.topRows(dim);
	dual_init = qp.solution.bottomRows(n_eq);

	Scalar eps_abs = Scalar(1e-10);
	auto iter = [&] {
		EigenNoAlloc _{};
		return qp::detail::solve_qp( //
				{from_eigen, primal_init},
				{from_eigen, dual_init},
				qp.as_view(),
				200,
				eps_abs,
				0,
				qp::preconditioner::IdentityPrecond{});
	}();

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);

	DOCTEST_CHECK(iter.n_iters == 0);
}
