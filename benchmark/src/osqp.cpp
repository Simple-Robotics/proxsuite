#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <osqp.h>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <ldlt/qp/eq_solver.hpp>
#include <ldlt/precond/ruiz.hpp>
#include <util.hpp>

using namespace ldlt;

using Scalar = double;

auto main() -> int {
	i32 dim = 1000;
	i32 n_eq = 100;

	double p = 1;
	auto cond = Scalar(1e2);

	auto H_eigen = ldlt_test::rand::sparse_positive_definite_rand(dim, cond, p);
	auto A_eigen = ldlt_test::rand::sparse_matrix_rand<Scalar>(n_eq, dim, p);
	auto g_eigen = ldlt_test::rand::vector_rand<Scalar>(dim);
	auto b_eigen = ldlt_test::rand::vector_rand<Scalar>(n_eq);
	i32 n_iter = 100;

	{
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp"); }
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "ours"); }
		(void)0;
	}
	{
		auto H = ldlt_test::eigen_to_osqp_mat(H_eigen);
		auto A = ldlt_test::eigen_to_osqp_mat(A_eigen);
		auto osqp = OSQPData{
				dim,
				n_eq,
				&H,
				&A,
				g_eigen.data(),
				b_eigen.data(),
				b_eigen.data(),
		};

		OSQPSettings osqp_settings{};
		osqp_set_default_settings(&osqp_settings);
		osqp_settings.eps_rel = 1e-9;
		osqp_settings.eps_abs = 1e-9;
		osqp_settings.warm_start = 0;
		osqp_settings.verbose = 0;

		OSQPWorkspace* osqp_work{};

		osqp_setup(&osqp_work, &osqp, &osqp_settings);

		LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp");
		for (i32 i = 0; i < n_iter; ++i) {
			{ osqp_solve(osqp_work); }
		}

		osqp_cleanup(osqp_work);
	}

	{
		Vec<Scalar> x(dim);
		Vec<Scalar> y(n_eq);

		auto H = Mat<Scalar, colmajor>(
				H_eigen.toDense().selfadjointView<Eigen::Upper>());
		auto A = Mat<Scalar, colmajor>(A_eigen.toDense());
		auto precond = qp::preconditioner::IdentityPrecond{};

		{
			LDLT_DECL_SCOPE_TIMER("osqp bench", "ours");
			for (i32 i = 0; i < n_iter; ++i) {
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
						1000,
						1e-9,
						0,
						LDLT_FWD(precond));
			};
		}
	}

	for (auto& outer : LDLT_GET_MAP()["osqp bench"]) {
		fmt::print("{}: {}\n", outer.first, outer.second.ref.back() / n_iter);
	}
}
