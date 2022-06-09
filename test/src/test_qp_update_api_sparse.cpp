#include <qp/sparse/wrapper.hpp>
#include <util.hpp>
#include <doctest.h>
#include <veg/util/dynstack_alloc.hpp>

using namespace qp;
using T = double;
using I = c_int;
using namespace linearsolver::sparse::tags;

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update rho") {
        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test update rho" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.settings.verbose = false;
        Qp.init(H,g,A,b,C,u,l);
        Qp.update_proximal_parameters(T(1.e-7), tl::nullopt, tl::nullopt);
        std::cout << "after upating" << std::endl;
        std::cout << "rho :  " << Qp.results.info.rho << std::endl;
        Qp.solve();
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update mus") {

        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test update mus" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 1.0;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.init(H,g,A,b,C,u,l);
        Qp.update_proximal_parameters(tl::nullopt, T(1.E-2), T(1.E-3));
        std::cout << "after upating" << std::endl;
        std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;
        std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;
        Qp.solve();

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}
TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with no equilibration at initialization") {

        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no equilibration at initialization" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.init(H,g,A,b,C,u,l,false);
        Qp.solve();

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with equilibration at initialization") {

        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test with equilibration at initialization" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.init(H,g,A,b,C,u,l,true);
        Qp.solve();

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}
TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with no initial guess") {

        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no initial guess" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::NO_INITIAL_GUESS;
        Qp.init(H,g,A,b,C,u,l);
        Qp.solve();

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update g for unconstrained problem") {

        std::cout << "------------------------sparse random strongly convex qp with equality and inequality constraints: test with no initial guess" << std::endl;
	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::NO_INITIAL_GUESS;
        Qp.init(H,g,A,b,C,u,l);
        Qp.solve();
        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

        g = ldlt_test::rand::vector_rand<T>(n);
        std::cout << "H before update " << H << std::endl;
        H *= 2.; // keep same sparsity structure
        std::cout << "H generated " << H << std::endl;
        Qp.update(
        H,
        g,
        A,
        b,
        C,
        u,
        l,false);
        linearsolver::sparse::MatMut<T, I> kkt_unscaled = Qp.model.kkt_mut_unscaled();
        auto kkt_top_n_rows = proxsuite::qp::sparse::detail::top_rows_mut_unchecked(veg::unsafe, kkt_unscaled, n);

	linearsolver::sparse::MatMut<T, I> H_unscaled = proxsuite::qp::sparse::detail::middle_cols_mut(kkt_top_n_rows, 0, n, Qp.model.H_nnz);
        std::cout << " H_unscaled " << H_unscaled.to_eigen() << std::endl;
        Qp.solve();

        dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
	}
}


TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting") {

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
        Qp.init(H,g,A,b,C,u,l);
        auto x_wm = ldlt_test::rand::vector_rand<T>(n);
        auto y_wm = ldlt_test::rand::vector_rand<T>(n_eq);
        auto z_wm = ldlt_test::rand::vector_rand<T>(n_in);
        std::cout << "proposed warm start" << std::endl;
        std::cout << "x_wm :  " << x_wm << std::endl;
        std::cout << "y_wm :  " << y_wm << std::endl;
        std::cout << "z_wm :  " << z_wm << std::endl;
        Qp.solve(x_wm,y_wm,z_wm);

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

	}
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with warm start with previous result") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test with warm start with previous result---"
						<< std::endl;

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();
                T eps_abs = 1.E-9;

                qp::sparse::QP<T,I> Qp(n, n_eq, n_in); // creating QP object
                Qp.settings.eps_abs =eps_abs;
                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
                Qp.init(H, g,
                                A, b,
                                C, u, l);
                Qp.solve();

                T pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                T dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

                qp::sparse::QP<T,I> Qp2(n, n_eq, n_in); // creating QP object
                Qp2.settings.eps_abs = 1.E-9;
                Qp2.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
                Qp2.init(H, g,
                                A, b,
                                C, u, l,true);
                
                auto x = Qp.results.x;
                auto y = Qp.results.y;
                auto z = Qp.results.z;
                //std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x << std::endl;
                Qp2.ruiz.scale_primal_in_place({proxsuite::qp::from_eigen,x});
                Qp2.ruiz.scale_dual_in_place_eq({proxsuite::qp::from_eigen,y});
                Qp2.ruiz.scale_dual_in_place_in({proxsuite::qp::from_eigen,z});
                //std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x << std::endl;
                Qp2.solve(x,y,z);

                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
                Qp.update(
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,true);
                Qp.solve();
                pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                std::cout << "------using API solving qp with dim with Qp after warm start with previous result: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
                pri_res = std::max(
                                (A * Qp2.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp2.results.x - u) +
                        sparse::detail::negative_part(C * Qp2.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y +
                        C.transpose() * Qp2.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp2: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp2.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;

        }
}

DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test with cold start option") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test with cold start option---"
						<< std::endl;

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

                T eps_abs = 1.E-9;
                qp::sparse::QP<T,I> Qp(n, n_eq, n_in); // creating QP object
                Qp.settings.eps_abs = eps_abs;
                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
                Qp.init(H, g,
                                A, b,
                                C, u, l);
                Qp.solve();

                T pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                T dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

                qp::sparse::QP<T,I> Qp2(n, n_eq, n_in); // creating QP object
                Qp2.settings.eps_abs = 1.E-9;
                Qp2.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
                Qp2.init(H, g,
                                A, b,
                                C, u, l);
                
                auto x = Qp.results.x;
                auto y = Qp.results.y;
                auto z = Qp.results.z;
                //std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x << std::endl;
                Qp2.ruiz.scale_primal_in_place({proxsuite::qp::from_eigen,x});
                Qp2.ruiz.scale_dual_in_place_eq({proxsuite::qp::from_eigen,y});
                Qp2.ruiz.scale_dual_in_place_in({proxsuite::qp::from_eigen,z});
                //std::cout << "after scaling x " << x <<  " Qp.results.x " << Qp.results.x << std::endl;
                Qp2.solve(x,y,z);

                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT;
                Qp.update(
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,true);
                Qp.solve();
                pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                std::cout << "------using API solving qp with dim with Qp after warm start with cold start option: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;
                pri_res = std::max(
                                (A * Qp2.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp2.results.x - u) +
                        sparse::detail::negative_part(C * Qp2.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y +
                        C.transpose() * Qp2.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with cold start option: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp2.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;

        }
}



DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test equilibration option") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test equilibration option---"
						<< std::endl;

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

                T eps_abs = 1.E-9;
                qp::sparse::QP<T,I> Qp(n, n_eq, n_in); // creating QP object
                Qp.settings.eps_abs = eps_abs;
                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
                Qp.init(H, g,
                                A, b,
                                C, u, l,true);
                Qp.solve();

                T pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                T dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

                qp::sparse::QP<T,I> Qp2(n, n_eq, n_in); // creating QP object
                Qp2.settings.eps_abs = 1.E-9;
                Qp2.init(H, g,
                                A, b,
                                C, u, l,false);
                Qp2.solve();
                pri_res = std::max(
                                (A * Qp2.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp2.results.x - u) +
                        sparse::detail::negative_part(C * Qp2.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y +
                        C.transpose() * Qp2.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp2: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp2.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;
        }
}


DOCTEST_TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test equilibration option at update") {

	std::cout << "---testing sparse random strongly convex qp with equality and "
	             "inequality constraints: test equilibration option at update---"
						<< std::endl;

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

                T eps_abs = 1.E-9;
                qp::sparse::QP<T,I> Qp(n, n_eq, n_in); // creating QP object
                Qp.settings.eps_abs = eps_abs;
                Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS;
                Qp.init(H, g,
                                A, b,
                                C, u, l,true);
                Qp.solve();
                T pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                T dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

                Qp.update(
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,true);
                Qp.solve();

                pri_res = std::max(
                                (A * Qp.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp.results.x - u) +
                        sparse::detail::negative_part(C * Qp.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y +
                        C.transpose() * Qp.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

                qp::sparse::QP<T,I> Qp2(n, n_eq, n_in); // creating QP object
                Qp2.settings.eps_abs = 1.E-9;
                Qp2.init(H, g,
                                A, b,
                                C, u, l,false);
                
                Qp2.solve();
                pri_res = std::max(
                                (A * Qp2.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp2.results.x - u) +
                        sparse::detail::negative_part(C * Qp2.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y +
                        C.transpose() * Qp2.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp2: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp2.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;

                Qp2.update(
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,
                                tl::nullopt,false);
                Qp2.solve();
                pri_res = std::max(
                                (A * Qp2.results.x - b).lpNorm<Eigen::Infinity>(),
                                (sparse::detail::positive_part(C * Qp2.results.x - u) +
                        sparse::detail::negative_part(C * Qp2.results.x - l))
                                                .lpNorm<Eigen::Infinity>());
                dua_res = (H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y +
                        C.transpose() * Qp2.results.z)
                                .lpNorm<Eigen::Infinity>();
                DOCTEST_CHECK(pri_res <= eps_abs);
                DOCTEST_CHECK(dua_res <= eps_abs);
                std::cout << "------using API solving qp with dim with Qp2: " << n
                                                        << " neq: " << n_eq << " nin: " << n_in << std::endl;
                std::cout << "primal residual: " << pri_res << std::endl;
                std::cout << "dual residual: " << dua_res << std::endl;
                std::cout << "total number of iteration: " << Qp2.results.info.iter
                                                        << std::endl;
                std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;
        }
}


TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test new init") {

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.settings.initial_guess = proxsuite::qp::InitialGuessStatus::WARM_START;
        Qp.init(H,g,A,b,C,u,l);
        auto x_wm = ldlt_test::rand::vector_rand<T>(n);
        auto y_wm = ldlt_test::rand::vector_rand<T>(n_eq);
        auto z_wm = ldlt_test::rand::vector_rand<T>(n_in);
        std::cout << "proposed warm start" << std::endl;
        std::cout << "x_wm :  " << x_wm << std::endl;
        std::cout << "y_wm :  " << y_wm << std::endl;
        std::cout << "z_wm :  " << z_wm << std::endl;
        Qp.solve(x_wm,y_wm,z_wm);

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;

	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test new init") {

	for (auto const& dims : {
					 //veg::tuplify(50, 0, 0),
					 //veg::tuplify(50, 25, 0),
					 //veg::tuplify(10, 0, 10),
					 //veg::tuplify(50, 0, 25),
					 //veg::tuplify(50, 10, 25),
                                         veg::tuplify(10, 2, 2)
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 0.15;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
                auto x_sol = ldlt_test::rand::vector_rand<T>(n);
	        auto b = A * x_sol;
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l =  C * x_sol; 
                auto u = (l.array() + 100).matrix().eval();

        qp::sparse::QP<T,I> Qp(H.cast<bool>(),A.cast<bool>(),C.cast<bool>());
        Qp.settings.eps_abs = 1.E-9;

        
        Qp.init(H,g,A,b,C,u,l);
        Qp.solve();

        T dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) ;
        T pri_res = std::max( qp::dense::infty_norm(A * Qp.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp.results.x - u) + sparse::detail::negative_part(C * Qp.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp.results.info.setup_time << " solve time " << Qp.results.info.solve_time << std::endl;


        qp::sparse::QP<T,I> Qp2(n,n_eq,n_in);
        Qp2.settings.eps_abs = 1.E-9;

        
        Qp2.init(H,g,A,b,C,u,l);
        Qp2.solve();

        dua_res = qp::dense::infty_norm(H.selfadjointView<Eigen::Upper>() * Qp2.results.x + g + A.transpose() * Qp2.results.y + C.transpose() * Qp2.results.z) ;
        pri_res = std::max( qp::dense::infty_norm(A * Qp2.results.x - b),
			qp::dense::infty_norm(sparse::detail::positive_part(C * Qp2.results.x - u) + sparse::detail::negative_part(C * Qp2.results.x - l)));
        CHECK(dua_res <= 1e-9);
        CHECK(pri_res <= 1E-9);
        std::cout << "--n = " << n << " n_eq " << n_eq << " n_in " << n_in << std::endl;
        std::cout  << "; dual residual " << dua_res << "; primal residual " <<  pri_res << std::endl;
	std::cout << "total number of iteration: " << Qp2.results.info.iter << std::endl;
	std::cout << "setup timing " << Qp2.results.info.setup_time << " solve time " << Qp2.results.info.solve_time << std::endl;

	}
}