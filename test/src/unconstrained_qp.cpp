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

DOCTEST_TEST_CASE(
		"sparse random strongly convex unconstrained qp and increasing dimension") {

	std::cout << "---testing sparse random strongly convex qp with increasing "
	             "dimension---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	for (isize dim = 10; dim < 1000; dim += 100) {

		isize n_eq(0);
		isize n_in(0);
		T strong_convexity_factor(1.e-2);
		Qp<T> qp{
				random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
		qp::QPSettings<T> settings;
		settings.eps_abs = eps_abs;
		settings.verbose = false;
		qp::QPData<T> data(dim, n_eq, n_in);
		qp::QPResults<T> results{dim, n_eq, n_in};
		qp::QPWorkspace<T> work{dim, n_eq, n_in};

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

		qp::detail::qp_solve(settings, data, results, work);
		T pri_res = std::max(
				(qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::detail::positive_part(qp.C * results.x - qp.u) +
		     qp::detail::negative_part(qp.C * results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
		             qp.C.transpose() * results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << results.n_tot << std::endl;
	}
}

DOCTEST_TEST_CASE("sparse random not strongly convex unconstrained qp and "
                  "increasing dimension") {

	std::cout << "---testing sparse random not strongly convex unconstrained qp "
	             "with increasing dimension---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	for (isize dim = 10; dim < 1000; dim += 100) {

		isize n_eq(0);
		isize n_in(0);
		T strong_convexity_factor(0);
		Qp<T> qp{
				random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
		auto x_sol = ldlt_test::rand::vector_rand<T>(dim);
		qp.g = -qp.H *
		       x_sol; // to be dually feasible g must be in the image space of H

		qp::QPSettings<T> settings;
		settings.eps_abs = eps_abs;
		settings.verbose = false;
		qp::QPData<T> data(dim, n_eq, n_in);
		qp::QPResults<T> results{dim, n_eq, n_in};
		qp::QPWorkspace<T> work{dim, n_eq, n_in};

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

		qp::detail::qp_solve(settings, data, results, work);
		T pri_res = std::max(
				(qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::detail::positive_part(qp.C * results.x - qp.u) +
		     qp::detail::negative_part(qp.C * results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
		             qp.C.transpose() * results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << results.n_tot << std::endl;
	}
}

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g random") {

	std::cout << "---unconstrained qp with H = Id and g random---" << std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);

	isize dim(100);
	isize n_eq(0);
	isize n_in(0);
	T strong_convexity_factor(1.E-2);
	Qp<T> qp{random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
	qp.H.setZero();
	qp.H.diagonal().array() += 1;

	qp::QPSettings<T> settings;
	settings.eps_abs = eps_abs;
	settings.verbose = false;
	qp::QPData<T> data(dim, n_eq, n_in);
	qp::QPResults<T> results{dim, n_eq, n_in};
	qp::QPWorkspace<T> work{dim, n_eq, n_in};

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

	qp::detail::qp_solve(settings, data, results, work);
	T pri_res = std::max(
			(qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(qp::detail::positive_part(qp.C * results.x - qp.u) +
	     qp::detail::negative_part(qp.C * results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
	             qp.C.transpose() * results.z)
	                .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
						<< " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << results.n_tot << std::endl;
}

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g = 0") {

	std::cout << "---unconstrained qp with H = Id and g = 0---" << std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);

	isize dim(100);
	isize n_eq(0);
	isize n_in(0);
	T strong_convexity_factor(1.E-2);
	Qp<T> qp{random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
	qp.H.setZero();
	qp.H.diagonal().array() += 1;
	qp.g.setZero();

	qp::QPSettings<T> settings;
	settings.eps_abs = eps_abs;
	settings.verbose = false;
	qp::QPData<T> data(dim, n_eq, n_in);
	qp::QPResults<T> results{dim, n_eq, n_in};
	qp::QPWorkspace<T> work{dim, n_eq, n_in};

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

	qp::detail::qp_solve(settings, data, results, work);
	T pri_res = std::max(
			(qp.A * results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(qp::detail::positive_part(qp.C * results.x - qp.u) +
	     qp::detail::negative_part(qp.C * results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	T dua_res = (qp.H * results.x + qp.g + qp.A.transpose() * results.y +
	             qp.C.transpose() * results.z)
	                .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
						<< " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << results.n_tot << std::endl;
}
