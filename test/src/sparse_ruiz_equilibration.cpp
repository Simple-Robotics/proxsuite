//
// Copyright (c) 2022 INRIA
//
#include <proxsuite/proxqp/sparse/solver.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz.hpp>
#include <util.hpp>
#include <doctest.h>
#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

using namespace proxqp;
using T = double;
using I = c_int;
using namespace proxsuite::linalg::sparse::tags;

TEST_CASE("upper part") {
	isize n = 10;
	isize n_eq = 6;
	isize n_in = 5;

	auto H = ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), 0.5);
	auto g = ldlt_test::rand::vector_rand<T>(n);
	auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_eq, 0.5);
	auto b = ldlt_test::rand::vector_rand<T>(n_eq);
	auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_in, 0.5);
	auto l = ldlt_test::rand::vector_rand<T>(n_in);
	auto u = ldlt_test::rand::vector_rand<T>(n_in);

	auto H_scaled = H;
	auto g_scaled = g;
	auto AT_scaled = AT;
	auto b_scaled = b;
	auto CT_scaled = CT;
	auto l_scaled = l;
	auto u_scaled = u;

	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> H_scaled_dense = H;
	auto g_scaled_dense = g;
	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> A_scaled_dense = AT.transpose();
	auto b_scaled_dense = b;
	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> C_scaled_dense = CT.transpose();
	auto l_scaled_dense = l;
	auto u_scaled_dense = u;

	proxqp::sparse::preconditioner::RuizEquilibration<T, I> ruiz{
			n,
			n_eq + n_in,
			1e-3,
			10,
			proxqp::sparse::preconditioner::Symmetry::UPPER,
	};
	proxqp::dense::preconditioner::RuizEquilibration<T> ruiz_dense{
			n,
			n_eq + n_in,
			1e-3,
			10,
			Symmetry::upper,
	};
	VEG_MAKE_STACK(
			stack, ruiz.scale_qp_in_place_req(proxsuite::linalg::veg::Tag<T>{}, n, n_eq, n_in));

	bool execute_preconditioner = true;
	proxsuite::proxqp::Settings<T> settings;

	ruiz.scale_qp_in_place(
			{
					{proxsuite::linalg::sparse::from_eigen, H_scaled},
					{proxsuite::linalg::sparse::from_eigen, g_scaled},
					{proxsuite::linalg::sparse::from_eigen, AT_scaled},
					{proxsuite::linalg::sparse::from_eigen, b_scaled},
					{proxsuite::linalg::sparse::from_eigen, CT_scaled},
					{proxsuite::linalg::sparse::from_eigen, l_scaled},
					{proxsuite::linalg::sparse::from_eigen, u_scaled},
			},
			execute_preconditioner,
			settings,
			stack);

	ruiz_dense.scale_qp_in_place(
			{
					{proxqp::from_eigen, H_scaled_dense},
					{proxqp::from_eigen, g_scaled_dense},
					{proxqp::from_eigen, A_scaled_dense},
					{proxqp::from_eigen, b_scaled_dense},
					{proxqp::from_eigen, C_scaled_dense},
					{proxqp::from_eigen, l_scaled_dense},
					{proxqp::from_eigen, u_scaled_dense},
			},
			execute_preconditioner,
			settings,
			stack);

	CHECK(H_scaled.toDense() == (H_scaled_dense));
	CHECK(g_scaled == (g_scaled_dense));
	CHECK(AT_scaled.transpose().toDense() == (A_scaled_dense));
	CHECK(b_scaled == (b_scaled_dense));
	CHECK(AT_scaled.transpose().toDense() == (A_scaled_dense));
	CHECK(l_scaled == (l_scaled_dense));
	CHECK(u_scaled == (u_scaled_dense));
}

TEST_CASE("lower part") {
	isize n = 3;
	isize n_eq = 0;
	isize n_in = 0;

	SparseMat<T> H =
			ldlt_test::rand::sparse_positive_definite_rand(n, T(10.0), 0.5)
					.transpose();
	auto g = ldlt_test::rand::vector_rand<T>(n);
	auto AT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_eq, 0.5);
	auto b = ldlt_test::rand::vector_rand<T>(n_eq);
	auto CT = ldlt_test::rand::sparse_matrix_rand<T>(n, n_in, 0.5);
	auto l = ldlt_test::rand::vector_rand<T>(n_in);
	auto u = ldlt_test::rand::vector_rand<T>(n_in);

	auto H_scaled = H;
	auto g_scaled = g;
	auto AT_scaled = AT;
	auto b_scaled = b;
	auto CT_scaled = CT;
	auto l_scaled = l;
	auto u_scaled = u;

	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> H_scaled_dense = H;
	auto g_scaled_dense = g;
	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> A_scaled_dense = AT.transpose();
	auto b_scaled_dense = b;
	Eigen::Matrix<T, -1, -1, Eigen::RowMajor> C_scaled_dense = CT.transpose();
	auto l_scaled_dense = l;
	auto u_scaled_dense = u;

	proxqp::sparse::preconditioner::RuizEquilibration<T, I> ruiz{
			n,
			n_eq + n_in,
			1e-3,
			10,
			proxqp::sparse::preconditioner::Symmetry::LOWER,
	};
	proxqp::dense::preconditioner::RuizEquilibration<T> ruiz_dense{
			n,
			n_eq + n_in,
			1e-3,
			10,
			Symmetry::lower,
	};
	VEG_MAKE_STACK(
			stack, ruiz.scale_qp_in_place_req(proxsuite::linalg::veg::Tag<T>{}, n, n_eq, n_in));
	bool execute_preconditioner = true;
	proxsuite::proxqp::Settings<T> settings;
	ruiz.scale_qp_in_place(
			{
					{proxsuite::linalg::sparse::from_eigen, H_scaled},
					{proxsuite::linalg::sparse::from_eigen, g_scaled},
					{proxsuite::linalg::sparse::from_eigen, AT_scaled},
					{proxsuite::linalg::sparse::from_eigen, b_scaled},
					{proxsuite::linalg::sparse::from_eigen, CT_scaled},
					{proxsuite::linalg::sparse::from_eigen, l_scaled},
					{proxsuite::linalg::sparse::from_eigen, u_scaled},
			},
			execute_preconditioner,
			settings,
			stack);

	ruiz_dense.scale_qp_in_place(
			{
					{proxqp::from_eigen, H_scaled_dense},
					{proxqp::from_eigen, g_scaled_dense},
					{proxqp::from_eigen, A_scaled_dense},
					{proxqp::from_eigen, b_scaled_dense},
					{proxqp::from_eigen, C_scaled_dense},
					{proxqp::from_eigen, l_scaled_dense},
					{proxqp::from_eigen, u_scaled_dense},
			},
			execute_preconditioner,
			settings,
			stack);

	CHECK(H_scaled.toDense() == (H_scaled_dense));
	CHECK(g_scaled == (g_scaled_dense));
	CHECK(AT_scaled.transpose().toDense() == (A_scaled_dense));
	CHECK(b_scaled == (b_scaled_dense));
	CHECK(AT_scaled.transpose().toDense() == (A_scaled_dense));
	CHECK(l_scaled == (l_scaled_dense));
	CHECK(u_scaled == (u_scaled_dense));
}
