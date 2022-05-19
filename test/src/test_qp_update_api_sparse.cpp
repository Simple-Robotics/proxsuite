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

	for (auto const& dims : {
					 veg::tuplify(50, 0, 0),
					 veg::tuplify(50, 25, 0),
					 veg::tuplify(10, 0, 10),
					 veg::tuplify(50, 0, 25),
					 veg::tuplify(50, 10, 25),
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 1.0;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l = ldlt_test::rand::vector_rand<T>(n_in);
		auto u = (l.array() + 1).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }
        Qp.cleanup();
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        Qp.update_proximal_parameter(T(1.e-7), tl::nullopt, tl::nullopt);
        std::cout << "after upating" << std::endl;
        std::cout << "rho :  " << Qp.results.info.rho << std::endl;

        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }

	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test update mus") {

	for (auto const& dims : {
					 veg::tuplify(50, 0, 0),
					 veg::tuplify(50, 25, 0),
					 veg::tuplify(10, 0, 10),
					 veg::tuplify(50, 0, 25),
					 veg::tuplify(50, 10, 25),
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 1.0;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l = ldlt_test::rand::vector_rand<T>(n_in);
		auto u = (l.array() + 1).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }
        Qp.cleanup();
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        Qp.update_proximal_parameter(tl::nullopt, T(1.E-2), T(1.E-3));
        std::cout << "after upating" << std::endl;
        std::cout << "mu_eq :  " << Qp.results.info.mu_eq << std::endl;
        std::cout << "mu_in :  " << Qp.results.info.mu_in << std::endl;

        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }

	}
}

TEST_CASE("sparse random strongly convex qp with equality and "
                  "inequality constraints: test warm starting") {

	for (auto const& dims : {
					 veg::tuplify(50, 0, 0),
					 veg::tuplify(50, 25, 0),
					 veg::tuplify(10, 0, 10),
					 veg::tuplify(50, 0, 25),
					 veg::tuplify(50, 10, 25),
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 1.0;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l = ldlt_test::rand::vector_rand<T>(n_in);
		auto u = (l.array() + 1).matrix().eval();

        qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
        Qp.settings.eps_abs = 1.E-9;
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        
        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }
        std::cout << "before upating" << std::endl;
        std::cout << "x :  " << Qp.results.x << std::endl;
        std::cout << "y :  " << Qp.results.y << std::endl;
        std::cout << "z :  " << Qp.results.z << std::endl;

        auto x_wm = ldlt_test::rand::vector_rand<T>(n);
        auto y_wm = ldlt_test::rand::vector_rand<T>(n_eq);
        auto z_wm = ldlt_test::rand::vector_rand<T>(n_in);
        std::cout << "proposed warm start" << std::endl;
        std::cout << "x_wm :  " << x_wm << std::endl;
        std::cout << "y_wm :  " << y_wm << std::endl;
        std::cout << "z_wm :  " << z_wm << std::endl;
        Qp.cleanup();
        Qp.setup_sparse_matrices(H,g,A,b,C,u,l);
        Qp.warm_start(x_wm,y_wm,z_wm);
        std::cout << "after update" << std::endl;
        std::cout << "x :  " << Qp.results.x << std::endl;
        std::cout << "y :  " << Qp.results.y << std::endl;
        std::cout << "z :  " << Qp.results.z << std::endl;
        Qp.solve();

        CHECK(
                qp::dense::infty_norm(
                        H.selfadjointView<Eigen::Upper>() * Qp.results.x + g + A.transpose() * Qp.results.y + C.transpose() * Qp.results.z) <=
                1e-9);
        CHECK(qp::dense::infty_norm(A * Qp.results.x - b) <= 1e-9);
        if (n_in > 0) {
            CHECK((C * Qp.results.x - l).minCoeff() > -1e-9);
            CHECK((C * Qp.results.x - u).maxCoeff() < 1e-9);
        }

	}
}

TEST_CASE("test API proper data and workspace cleanup") {

	for (auto const& dims : {
					 veg::tuplify(2, 0, 2),
					 veg::tuplify(50, 0, 0),
					 veg::tuplify(50, 25, 0),
					 veg::tuplify(50, 0, 25),
					 veg::tuplify(50, 10, 25),
			 }) {
		VEG_BIND(auto const&, (n, n_eq, n_in), dims);

		double p = 1.0;

		auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), p);
		auto g = ldlt_test::rand::vector_rand<T>(n);
		auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n,n_eq, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n,n_in, p);
		auto l = ldlt_test::rand::vector_rand<T>(n_in);
		auto u = (l.array() + 1).matrix().eval();

		{

			sparse::QpView<T, I> qp = {
					{linearsolver::sparse::from_eigen, H},
					{linearsolver::sparse::from_eigen, g},
					{linearsolver::sparse::from_eigen, AT},
					{linearsolver::sparse::from_eigen, b},
					{linearsolver::sparse::from_eigen, CT},
					{linearsolver::sparse::from_eigen, l},
					{linearsolver::sparse::from_eigen, u},
			};

			sparse::Workspace<T, I> work;
			sparse::Model<T, I> data;

            for (isize i = 0; i < 10; i++) {

                qp::sparse::preconditioner::RuizEquilibration<T, I> ruiz{
                        n,
                        n_eq + n_in,
                        1e-3,
                        10,
                        qp::sparse::preconditioner::Symmetry::UPPER,
                };
                Settings<T> settings;
                Results<T> results;
                
                sparse::qp_setup(qp, results, data, work, ruiz);

                auto& x = results.x;
                auto& y = results.y;
                auto& z = results.z;

                x.setZero();
                y.setZero();
                z.setZero();

                sparse::qp_solve(results, data, settings, work, ruiz);
                CHECK(
                        qp::dense::infty_norm(
                                H.selfadjointView<Eigen::Upper>() * results.x + g + AT * results.y + CT * results.z) <=
                        1e-9);
                CHECK(qp::dense::infty_norm(AT.transpose() * results.x - b) <= 1e-9);
                if (n_in > 0) {
                    CHECK((CT.transpose() * results.x - l).minCoeff() > -1e-9);
                    CHECK((CT.transpose() * results.x - u).maxCoeff() < 1e-9);
                }
            }

		}
	}
}