#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/solver.hpp>
#include <qp/dense/precond/ruiz.hpp>
#include <qp/dense/precond/identity.hpp>
#include <veg/util/dbg.hpp>
#include <util.hpp>

using namespace ldlt;

using T = double;

DOCTEST_TEST_CASE("qp: start from solution") {
	isize dim = 30;
	isize n_eq = 6;
	isize n_in = 0;
	std::cout << "---testing sparse random strongly convex qp with equality constraints and starting at the solution---" << std::endl;

	Qp<T> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<T, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init = qp.solution.topRows(dim);
	dual_init = qp.solution.bottomRows(n_eq);

	T eps_abs = T(1e-9);

	qp::Settings<T> settings;
	qp::dense::Data<T> data(dim, n_eq, n_in);
	qp::Results<T> results{dim, n_eq, n_in};
	qp::dense::Workspace<T> work{dim, n_eq, n_in};

	results.x = primal_init;
	results.y = dual_init;

	qp::dense::QPsetup_dense<T>( //
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
		qp::dense::qp_solve(settings, data, results, work);
	}

	DOCTEST_CHECK((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * results.x + qp.g + qp.A.transpose() * results.y)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}


DOCTEST_TEST_CASE("sparse random strongly convex qp with equality constraints and increasing dimension") {

	std::cout << "---testing sparse random strongly convex qp with equality constraints and increasing dimension---" << std::endl;
	double  sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	for (isize dim = 10; dim < 1000; dim+=100) {

		isize n_eq (dim / 2);
		isize n_in (0);
		T strong_convexity_factor(1.e-2);
		Qp<T> qp{random_with_dim_and_neq_and_n_in, dim, n_eq, n_in, sparsity_factor, strong_convexity_factor};
		qp::Settings<T> settings;
		settings.eps_abs = eps_abs;
		settings.verbose = false;
		qp::dense::Data<T> data(dim, n_eq, n_in);
		qp::Results<T> results{dim, n_eq, n_in};
		qp::dense::Workspace<T> work{dim, n_eq, n_in};

		qp::dense::QPsetup_dense<T>( //
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

		qp::dense::qp_solve(settings, data, results, work);
		T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * results.x - qp.u) + qp::dense::negative_part(qp.C * results.x - qp.l) ).lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y + qp.C.transpose() * results.z).lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK( pri_res <= eps_abs);
		DOCTEST_CHECK( dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: "  << dua_res << std::endl;
		std::cout << "total number of iteration: " << results.n_tot << std::endl;
	}
}

DOCTEST_TEST_CASE("linear problem with equality  with equality constraints and linar cost and increasing dimension") {

	std::cout << "---testing linear problem with equality constraints and increasing dimension---" << std::endl;
	double  sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	for (isize dim = 10; dim < 1000; dim+=100) {

		isize n_eq (dim / 2);
		isize n_in (0);
		T strong_convexity_factor(1.e-2);
		Qp<T> qp{random_with_dim_and_neq_and_n_in, dim, n_eq, n_in, sparsity_factor, strong_convexity_factor};
		qp.H.setZero();
		auto y_sol = ldlt_test::rand::vector_rand<T>(n_eq); // make sure the LP is bounded within the feasible set
		qp.g = - qp.A.transpose() * y_sol ;
		qp::Settings<T> settings;
		settings.eps_abs = eps_abs;
		settings.verbose = false;
		qp::dense::Data<T> data(dim, n_eq, n_in);
		qp::Results<T> results{dim, n_eq, n_in};
		qp::dense::Workspace<T> work{dim, n_eq, n_in};

		qp::dense::QPsetup_dense<T>( //
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

		qp::dense::qp_solve(settings, data, results, work);
		T pri_res = std::max((qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * results.x - qp.u) + qp::dense::negative_part(qp.C * results.x - qp.l) ).lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y + qp.C.transpose() * results.z).lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK( pri_res <= eps_abs);
		DOCTEST_CHECK( dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: "  << dua_res << std::endl;
		std::cout << "total number of iteration: " << results.n_tot << std::endl;
	}
}