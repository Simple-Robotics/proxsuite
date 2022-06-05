#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <veg/util/dbg.hpp>
#include <util.hpp>

using T = double;

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update H") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update H---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::dense::isize dim = 10;

	proxsuite::qp::dense::isize n_eq(dim / 4);
	proxsuite::qp::dense::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating H" << std::endl;
	qp.H.setIdentity();
	Qp.update(
			qp.H,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);
	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update A") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update A---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;

	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating A" << std::endl;
	qp.A = ldlt_test::rand::sparse_matrix_rand_not_compressed<T>(
			n_eq, dim, sparsity_factor);
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			qp.A,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update C") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update C---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;

	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating C" << std::endl;
	qp.C = ldlt_test::rand::sparse_matrix_rand_not_compressed<T>(
			n_in, dim, sparsity_factor);
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			qp.C,
			tl::nullopt,
			tl::nullopt);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update b") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update b---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;

	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating b" << std::endl;
	auto x_sol = ldlt_test::rand::vector_rand<T>(dim);
	qp.b = qp.A * x_sol;
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			qp.b,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	     	 proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update u") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update u---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating b" << std::endl;
	auto x_sol = ldlt_test::rand::vector_rand<T>(dim);
	auto delta = Vec<T>(n_in);
	for (proxsuite::qp::isize i = 0; i < n_in; ++i) {
		delta(i) = ldlt_test::rand::uniform_rand();
	}

	qp.u = qp.C * x_sol + delta;
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			qp.u,
			tl::nullopt);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update g") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update g---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating g" << std::endl;
	auto g = ldlt_test::rand::vector_rand<T>(dim);

	qp.g = g;
	Qp.update(
			tl::nullopt,
			qp.g,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE(
		"sparse random strongly convex qp with equality and inequality "
    "constraints: test update H and A and b and u and l") {

	std::cout
			<< "---testing sparse random strongly convex qp with equality and "
	       "inequality constraints: test update H and A and b and u and l---"
			<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "H :  " << qp.H << std::endl;
	std::cout << "g :  " << qp.g << std::endl;
	std::cout << "A :  " << qp.A << std::endl;
	std::cout << "b :  " << qp.b << std::endl;
	std::cout << "C :  " << qp.C << std::endl;
	std::cout << "u :  " << qp.u << std::endl;
	std::cout << "l :  " << qp.l << std::endl;

	std::cout << "testing updating b" << std::endl;
	qp.H = ldlt_test::rand::sparse_positive_definite_rand_not_compressed<T>(
			dim, strong_convexity_factor, sparsity_factor);
	qp.A = ldlt_test::rand::sparse_matrix_rand_not_compressed<T>(
			n_eq, dim, sparsity_factor);
	auto x_sol = ldlt_test::rand::vector_rand<T>(dim);
	auto delta = Vec<T>(n_in);
	for (qp::isize i = 0; i < n_in; ++i) {
		delta(i) = ldlt_test::rand::uniform_rand();
	}
	qp.b = qp.A * x_sol;
	qp.u = qp.C * x_sol + delta;
	qp.l = qp.C * x_sol - delta;
	Qp.update(qp.H, tl::nullopt, qp.A, qp.b, tl::nullopt, qp.u, qp.l);

	std::cout << "after upating" << std::endl;
	std::cout << "H :  " << Qp.model.H << std::endl;
	std::cout << "g :  " << Qp.model.g << std::endl;
	std::cout << "A :  " << Qp.model.A << std::endl;
	std::cout << "b :  " << Qp.model.b << std::endl;
	std::cout << "C :  " << Qp.model.C << std::endl;
	std::cout << "u :  " << Qp.model.u << std::endl;
	std::cout << "l :  " << Qp.model.l << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update rho") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update rho---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "rho :  " << Qp.results.info.rho << std::endl;

	
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt); // restart the problem with default options
	Qp.update_proximal_parameters(T(1.e-7), tl::nullopt, tl::nullopt); // update one parameter of the problem
	std::cout << "after upating" << std::endl;
	std::cout << "rho :  " << Qp.results.info.rho << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update mu_eq and mu_in") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test update mu_eq and mu_in---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	std::cout << "before upating" << std::endl;
	std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
	std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;

	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);
	Qp.update_proximal_parameters(
			tl::nullopt, T(1.e-2), T(1.e-3)); // after update should redo a setup
	std::cout << "after upating" << std::endl;
	std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
	std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;

	Qp.solve();

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test warm starting---"
						<< std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);
	ldlt_test::rand::set_seed(1);
	proxsuite::qp::isize dim = 10;
	proxsuite::qp::isize n_eq(dim / 4);
	proxsuite::qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor};

	proxsuite::qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
	Qp.solve();

	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
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

	auto x_wm = ldlt_test::rand::vector_rand<T>(dim);
	auto y_wm = ldlt_test::rand::vector_rand<T>(n_eq);
	auto z_wm = ldlt_test::rand::vector_rand<T>(n_in);
	std::cout << "proposed warm start" << std::endl;
	std::cout << "x_wm :  " << x_wm << std::endl;
	std::cout << "y_wm :  " << y_wm << std::endl;
	std::cout << "z_wm :  " << z_wm << std::endl;
	Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
	Qp.update(
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt,
			tl::nullopt);
	Qp.solve(x_wm, y_wm, z_wm);

	pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(proxsuite::qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
	         proxsuite::qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
	           qp.C.transpose() * Qp.results.z)
	              .lpNorm<Eigen::Infinity>();
	DOCTEST_CHECK(pri_res <= eps_abs);
	DOCTEST_CHECK(dua_res <= eps_abs);

	std::cout << "------using API solving qp with dim after updating: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}
