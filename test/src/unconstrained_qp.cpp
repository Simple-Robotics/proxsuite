//
// Copyright (c) 2022, INRIA
//
#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <veg/util/dbg.hpp>
#include <util.hpp>

using namespace qp;

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
		Qp<T> qp{random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
		qp::dense::QP<T> Qp{dim,n_eq,n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.init(qp.H,qp.g,qp.A,qp.b,qp.C,qp.u,qp.l);
		Qp.solve();

		T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * Qp.results.x - qp.u) + qp::dense::negative_part(qp.C * Qp.results.x - qp.l) ).lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z).lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK( pri_res <= eps_abs);
		DOCTEST_CHECK( dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: "  << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
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

		qp::dense::QP<T> Qp{dim,n_eq,n_in}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.init(qp.H,qp.g,qp.A,qp.b,qp.C,qp.u,qp.l);
		Qp.solve();

		T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * Qp.results.x - qp.u) + qp::dense::negative_part(qp.C * Qp.results.x - qp.l) ).lpNorm<Eigen::Infinity>());
		T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z).lpNorm<Eigen::Infinity>();
		DOCTEST_CHECK( pri_res <= eps_abs);
		DOCTEST_CHECK( dua_res <= eps_abs);

		std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
		std::cout << "primal residual: " << pri_res << std::endl;
		std::cout << "dual residual: "  << dua_res << std::endl;
		std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	}
}

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g random") {

	std::cout << "---unconstrained qp with H = Id and g random---" << std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);

    isize dim (100);
    isize n_eq (0);
    isize n_in (0);
    T strong_convexity_factor(1.E-2);
    Qp<T> qp{random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
    qp.H.setZero();
    qp.H.diagonal().array()+= 1;

	qp::dense::QP<T> Qp{dim,n_eq,n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H,qp.g,qp.A,qp.b,qp.C,qp.u,qp.l);
	Qp.solve();

	T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * Qp.results.x - qp.u) + qp::dense::negative_part(qp.C * Qp.results.x - qp.l) ).lpNorm<Eigen::Infinity>());
	T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z).lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK( pri_res <= eps_abs);
    DOCTEST_CHECK( dua_res <= eps_abs);

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: "  << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;

}

DOCTEST_TEST_CASE("unconstrained qp with H = Id and g = 0") {

	std::cout << "---unconstrained qp with H = Id and g = 0---" << std::endl;
	double sparsity_factor = 0.15;
	T eps_abs = T(1e-9);

    isize dim (100);
    isize n_eq (0);
    isize n_in (0);
    T strong_convexity_factor(1.E-2);
    Qp<T> qp{random_unconstrained, dim, sparsity_factor, strong_convexity_factor};
    qp.H.setZero();
    qp.H.diagonal().array()+= 1;
    qp.g.setZero();

	qp::dense::QP<T> Qp{dim,n_eq,n_in}; // creating QP object
	Qp.settings.eps_abs = eps_abs;
	Qp.init(qp.H,qp.g,qp.A,qp.b,qp.C,qp.u,qp.l);
	Qp.solve();

	T pri_res = std::max((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(), (qp::dense::positive_part(qp.C * Qp.results.x - qp.u) + qp::dense::negative_part(qp.C * Qp.results.x - qp.l) ).lpNorm<Eigen::Infinity>());
	T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y + qp.C.transpose() * Qp.results.z).lpNorm<Eigen::Infinity>();
    DOCTEST_CHECK( pri_res <= eps_abs);
    DOCTEST_CHECK( dua_res <= eps_abs);

    std::cout << "------solving qp with dim: " << dim << " neq: " << n_eq  << " nin: " << n_in << std::endl;
    std::cout << "primal residual: " << pri_res << std::endl;
    std::cout << "dual residual: "  << dua_res << std::endl;
    std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;

}
