#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>

#include <ldlt/qp/eq_solver.hpp>
#include <ldlt/precond/ruiz.hpp>
#include <util.hpp>

#include <osqp.h>

using namespace ldlt;

using Scalar = c_float;

auto main() -> int {
	i32 dim = 1000;
	i32 n_eq = 100;

	double p = 0.15;
	auto cond = Scalar(1e2);

	auto H_eigen = ldlt_test::rand::sparse_positive_definite_rand(dim, cond, p);
	auto A_eigen = ldlt_test::rand::sparse_matrix_rand<Scalar>(n_eq, dim, p);
	auto g_eigen = ldlt_test::rand::vector_rand<Scalar>(dim);
	auto b_eigen = ldlt_test::rand::vector_rand<Scalar>(n_eq);
	i32 n_runs = 100;

	{
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp"); }
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "ours"); }
		(void)0;
	}

	Vec<Scalar> x(dim);
	Vec<Scalar> y(n_eq);

	i32 max_iter = 1000;
	Scalar eps_abs = Scalar(1e-9);
	Scalar eps_rel = 0;

	{
		LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp");
		ldlt_test::osqp::solve_eq_osqp_sparse(
				detail::from_eigen_vector_mut(x),
				detail::from_eigen_vector_mut(y),
				H_eigen,
				A_eigen,
				detail::from_eigen_vector(g_eigen),
				detail::from_eigen_vector(b_eigen),
				max_iter,
				eps_abs,
				eps_rel);
	}

	{

		auto H = Mat<Scalar, colmajor>(
				H_eigen.toDense().selfadjointView<Eigen::Upper>());
		auto A = Mat<Scalar, colmajor>(A_eigen.toDense());
		auto precond = qp::preconditioner::IdentityPrecond{};

		{
			LDLT_DECL_SCOPE_TIMER("osqp bench", "ours");
			for (i32 i = 0; i < n_runs; ++i) {
				x.setZero();
				y.setZero();

				qp::detail::solve_qp(
						VectorViewMut<Scalar>{x.data(), i32(x.rows())},
						VectorViewMut<Scalar>{y.data(), i32(y.rows())},
						qp::QpView<Scalar, colmajor, colmajor>{
								detail::from_eigen_matrix(H),
								detail::from_eigen_vector(g_eigen),
								detail::from_eigen_matrix(A),
								detail::from_eigen_vector(b_eigen),
								{},
								{},
						},
						max_iter,
						eps_abs,
						eps_rel,
						LDLT_FWD(precond));
			};
		}
	}

	for (auto& outer : LDLT_GET_MAP()["osqp bench"]) {
		fmt::print("{}: {}\n", outer.first, outer.second.ref.back() / n_runs);
	}
}
