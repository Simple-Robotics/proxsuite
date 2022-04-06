#include <qp/proxqp/solver_sparse.hpp>
#include <qp/precond/ruiz.hpp>
#include <util.hpp>
#include <doctest.h>
#include <veg/util/dynstack_alloc.hpp>

using namespace qp;
using T = double;
using I = c_int;
using namespace sparse_ldlt::tags;

TEST_CASE("qp random test") {
	isize n = 3;
	isize n_eq = 1;
	isize n_in = 2;

	auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), 0.5);
	auto g = ldlt_test::rand::vector_rand<T>(n);
	auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_eq, 0.5);
	auto b = ldlt_test::rand::vector_rand<T>(n_eq);
	auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_in, 0.5);
	auto l = ldlt_test::rand::vector_rand<T>(n_in);
	auto u = ldlt_test::rand::vector_rand<T>(n_in);

	qp::sparse::preconditioner::RuizEquilibration<T, I> ruiz{
			n,
			n_eq + n_in,
			1e-3,
			10,
			qp::sparse::preconditioner::Symmetry::UPPER,
	};

	sparse::QpView<T, I> qp = {
			{sparse_ldlt::from_eigen, H},
			{sparse_ldlt::from_eigen, g},
			{sparse_ldlt::from_eigen, AT},
			{sparse_ldlt::from_eigen, b},
			{sparse_ldlt::from_eigen, CT},
			{sparse_ldlt::from_eigen, l},
			{sparse_ldlt::from_eigen, u},
	};

	sparse::QpWorkspace<T, I> work;
	QPSettings<T> settings;
	sparse::qp_setup(work, qp);

	Eigen::Matrix<T, -1, 1> x(n);
	Eigen::Matrix<T, -1, 1> y(n_eq);
	Eigen::Matrix<T, -1, 1> z(n_in);
	x.setZero();
	y.setZero();
	z.setZero();

	sparse::qp_solve(
			{ldlt::from_eigen, x},
			{ldlt::from_eigen, y},
			{ldlt::from_eigen, z},
			work,
			ruiz,
			settings,
			qp);
}
