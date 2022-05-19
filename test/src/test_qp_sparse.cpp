#include <qp/sparse/wrapper.hpp>
#include <util.hpp>
#include <doctest.h>
#include <veg/util/dynstack_alloc.hpp>

using namespace qp;
using T = double;
using I = c_int;
using namespace linearsolver::sparse::tags;
/*
TEST_CASE("random ruiz") {

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
		auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_eq, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_in, p);
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

			qp::sparse::preconditioner::RuizEquilibration<T, I> ruiz{
					n,
					n_eq + n_in,
					1e-3,
					10,
					qp::sparse::preconditioner::Symmetry::UPPER,
			};

			sparse::Workspace<T, I> work;
			Settings<T> settings;
			Results<T> results;
			sparse::Data<T, I> data;
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
							H.selfadjointView<Eigen::Upper>() * x + g + AT * y + CT * z) <=
					1e-9);
			CHECK(qp::dense::infty_norm(AT.transpose() * x - b) <= 1e-9);
			if (n_in > 0) {
				CHECK((CT.transpose() * x - l).minCoeff() > -1e-9);
				CHECK((CT.transpose() * x - u).maxCoeff() < 1e-9);
			}
		}
	}
}

TEST_CASE("random ruiz using the API") {

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
		auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
		auto l = ldlt_test::rand::vector_rand<T>(n_in);
		auto u = (l.array() + 1).matrix().eval();

		{

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
		}
	}
}

TEST_CASE("random id") {

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
		auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_eq, p);
		auto b = ldlt_test::rand::vector_rand<T>(n_eq);
		auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_in, p);
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

			qp::sparse::preconditioner::Identity<T, I> id;

			sparse::Workspace<T, I> work;
			Settings<T> settings;
			Results<T> results;
			sparse::Data<T, I> data;
			sparse::qp_setup(qp, results, data, work, id);

			auto& x = results.x;
			auto& y = results.y;
			auto& z = results.z;
			x.setZero();
			y.setZero();
			z.setZero();

			sparse::qp_solve(results, data, settings, work, id);

			CHECK(
					qp::dense::infty_norm(
							H.selfadjointView<Eigen::Upper>() * x + g + AT * y + CT * z) <=
					1e-9);
			CHECK(qp::dense::infty_norm(AT.transpose() * x - b) <= 1e-9);
			if (n_in > 0) {
				CHECK((CT.transpose() * x - l).minCoeff() > -1e-9);
				CHECK((CT.transpose() * x - u).maxCoeff() < 1e-9);
			}
		}
	}
}
*/
TEST_CASE("random id using the API") {

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

		{
			qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
			Qp.settings.eps_abs = 1.E-9;
			Qp.settings.verbose = true;
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
		}
	}
}
