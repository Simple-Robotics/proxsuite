#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <veg/util/dbg.hpp>
#include <util.hpp>

using T = double;

DOCTEST_TEST_CASE(
		"sparse random strongly convex qp with equality and inequality constraints "
    "and increasing dimension using wrapper API") {

	std::cout
			<< "---testing sparse random strongly convex qp with equality and "
	       "inequality constraints and increasing dimension using wrapper API---"
			<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	for (qp::isize dim = 10; dim < 1000; dim += 100) {

		qp::isize n_eq(dim / 4);
		qp::isize n_in(dim / 4);
		T strong_convexity_factor(1.e-2);
		Qp<T> qp{
				random_with_dim_and_neq_and_n_in,
				dim,
				n_eq,
				n_in,
				sparsity_factor,
				strong_convexity_factor};

		qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
		Qp.solve();

		T pri_res = std::max(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
		     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
		             qp.C.transpose() * Qp.results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------using API solving qp with dim: " << dim
							<< " neq: " << n_eq << " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter
							<< std::endl;
	}
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with box inequality "
                  "constraints and increasing dimension using the API") {

	std::cout
			<< "---testing sparse random strongly convex qp with box inequality "
	       "constraints and increasing dimension using the API---"
			<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	for (qp::isize dim = 10; dim < 1000; dim += 100) {

		qp::isize n_eq(0);
		qp::isize n_in(dim);
		T strong_convexity_factor(1.e-2);
		Qp<T> qp{
				random_with_dim_and_n_in_and_box_constraints,
				dim,
				sparsity_factor,
				strong_convexity_factor};
		qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
		Qp.solve();
		T pri_res = std::max(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
		     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
		             qp.C.transpose() * Qp.results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter
							<< std::endl;
	}
}

DOCTEST_TEST_CASE("sparse random not strongly convex qp with inequality "
                  "constraints and increasing dimension using the API") {

	std::cout
			<< "---testing sparse random not strongly convex qp with inequality "
	       "constraints and increasing dimension using the API---"
			<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	for (qp::isize dim = 10; dim < 1000; dim += 100) {
		qp::isize n_in(dim / 2);
		qp::isize n_eq(0);
		Qp<T> qp{
				random_with_dim_and_n_in_not_strongly_convex,
				dim,
				n_in,
				sparsity_factor};
		qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
		Qp.solve();
		T pri_res = std::max(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
		     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
		             qp.C.transpose() * Qp.results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter
							<< std::endl;
	}
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with degenerate inequality "
                  "constraints and increasing dimension using the API") {

	std::cout
			<< "---testing sparse random strongly convex qp with degenerate "
	       "inequality constraints and increasing dimension using the API---"
			<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	T strong_convexity_factor(1e-2);
	ldlt_test::rand::set_seed(1);
	for (qp::isize dim = 10; dim < 1000; dim += 100) {
		qp::isize m(dim / 4);
		qp::isize n_in(2 * m);
		qp::isize n_eq(0);
		Qp<T> qp{
				random_with_dim_and_n_in_degenerate,
				dim,
				m,
				sparsity_factor,
				strong_convexity_factor};
		qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
		Qp.solve();
		T pri_res = std::max(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
		     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
		             qp.C.transpose() * Qp.results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter
							<< std::endl;
	}
}

DOCTEST_TEST_CASE("linear problem with equality inequality constraints and "
                  "increasing dimension using the API") {
	srand(1);
	std::cout << "---testing linear problem with inequality constraints and "
	             "increasing dimension using the API---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	for (qp::isize dim = 10; dim < 1000; dim += 100) {
		qp::isize n_in(dim / 2);
		qp::isize n_eq(0);
		Qp<T> qp{
				random_with_dim_and_n_in_not_strongly_convex,
				dim,
				n_in,
				sparsity_factor};
		qp.H.setZero();
		auto z_sol = ldlt_test::rand::vector_rand<T>(n_in);
		qp.g = -qp.C.transpose() *
		       z_sol; // make sure the LP is bounded within the feasible set
		//std::cout << "g : " << qp.g << " C " << qp.C  << " u " << qp.u << " l " << qp.l << std::endl;
		qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.settings.verbose = false;
		Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
		Qp.solve();
		T pri_res = std::max(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
		     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
						.lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
		             qp.C.transpose() * Qp.results.z)
		                .lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK(pri_res <= eps_abs);
		DOCTEST_CHECK(dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq
							<< " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: " << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter
							<< std::endl;
	}
}
