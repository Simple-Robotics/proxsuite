#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/proxqp/solver.hpp>
#include <qp/precond/ruiz.hpp>
#include <qp/precond/identity.hpp>
#include <veg/util/dbg.hpp>
#include <util.hpp>

using namespace ldlt;

using T = double;

DOCTEST_TEST_CASE("qp: start from solution") {
	isize dim = 30;
	isize n_eq = 6;
	isize n_in = 0;

	Qp<T> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init = qp.solution.topRows(dim);
	dual_init = qp.solution.bottomRows(n_eq);

	T eps_abs = T(1e-10);

	qp::QPSettings<T> settings;
	qp::QPData<T> data(dim, n_eq, n_in);
	qp::QPResults<T> results{dim, n_eq, n_in};
	qp::QPWorkspace<T> work{dim, n_eq, n_in};

	results.x = primal_init;
	results.y = dual_init;

	qp::detail::QPsetup_dense<T>( //
			qp.H,
			qp.g,
			qp.A,
			qp.b,
			qp.C,
			qp.u,
			qp.l,
			settings,
			data,
			work,
			results);
	{
		EigenNoAlloc _{};
		qp::detail::qp_solve(settings, data, results, work);
	}

	DOCTEST_CHECK((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * results.x + qp.g + qp.A.transpose() * results.y)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}
