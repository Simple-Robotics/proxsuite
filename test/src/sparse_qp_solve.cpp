//
// Copyright (c) 2022, INRIA
//
#include <doctest.h>
#include <proxsuite/qp/sparse/sparse.hpp>
#include <util.hpp>
#include <proxsuite/veg/util/dynstack_alloc.hpp>

using namespace qp;
using T = double;
using I = c_int;
using namespace linearsolver::sparse::tags;

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve function") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test solve function---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
        ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
        auto x_sol = ldlt_test::rand::vector_rand<T>(n);
        auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        std::nullopt,std::nullopt,std::nullopt,
                                                                        eps_abs);

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;

    }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve with different rho value") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test solve with different rho value---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
                ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
            auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        std::nullopt,std::nullopt,std::nullopt,
                                                                        eps_abs,std::nullopt,T(1.E-7));
        DOCTEST_CHECK(results.info.rho == T(1.E-7));
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;
    }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test solve with different mu_eq and mu_in values") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test solve with different mu_eq and mu_in values---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
                ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
            auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        std::nullopt,std::nullopt,std::nullopt,
                                                                        eps_abs,std::nullopt,std::nullopt,T(1.E-2),T(1.E-2));
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;
    }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test warm starting---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
                ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
        auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        auto x_wm = ldlt_test::rand::vector_rand<T>(n);
        auto y_wm = ldlt_test::rand::vector_rand<T>(n_eq);
        auto z_wm = ldlt_test::rand::vector_rand<T>(n_in);
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        x_wm,y_wm,z_wm,eps_abs);
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;
    }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test verbose = true") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test verbose = true ---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
                ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
        auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        bool verbose = true;
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        std::nullopt,std::nullopt,std::nullopt,eps_abs,
                                                                        std::nullopt,std::nullopt,std::nullopt,std::nullopt,
                                                                        verbose);
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;
    }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test no initial guess") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test no initial guess ---"
						<< std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 3, 2)
			 }) {
        VEG_BIND(auto const&, (n, n_eq, n_in), dims);
                
        double eps_abs = 1.e-9;
        double p = 0.15;
                ldlt_test::rand::set_seed(1);
        auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
        auto g = ldlt_test::rand::vector_rand<T>(n);
        auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
        auto b = A * x_sol;
        auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
        auto l =  C * x_sol; 
        auto u = (l.array() + 10).matrix().eval();
        proxsuite::qp::InitialGuessStatus initial_guess = proxsuite::qp::InitialGuessStatus::NO_INITIAL_GUESS;
        proxsuite::qp::Results<T> results = proxsuite::qp::sparse::solve<T,I>(H, g, A, b, C, u, l,
                                                                        std::nullopt,std::nullopt,std::nullopt,eps_abs,
                                                                        std::nullopt,std::nullopt,std::nullopt,std::nullopt,
                                                                        std::nullopt,true,true,std::nullopt,initial_guess);
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * results.x + g + A.transpose() * results.y + C.transpose() * results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * results.x - u) + sparse::detail::negative_part(C * results.x - l)));
        DOCTEST_CHECK(pri_res <= eps_abs);
        DOCTEST_CHECK(dua_res <= eps_abs);

        std::cout << "------using API solving qp with dim: " << n
                            << " neq: " << n_eq << " nin: " << n_in << std::endl;
        std::cout << "primal residual: " << pri_res << std::endl;
        std::cout << "dual residual: " << dua_res << std::endl;
        std::cout << "total number of iteration: " << results.info.iter
                            << std::endl;
        std::cout << "setup timing " << results.info.setup_time << " solve time " << results.info.solve_time << std::endl;
    }
}
