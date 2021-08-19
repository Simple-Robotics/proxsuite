#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <ldlt/qp_eq.hpp>
#include <fmt/core.h>

using namespace ldlt;

LDLT_DEFINE_TAG(with_dim_and_n_eq, WithDimAndNeq);
LDLT_DEFINE_TAG(random_with_dim_and_n_eq, RandomWithDimAndNeq);

template <typename Scalar>
struct Qp {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> H;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> g;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> b;

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> solution;

	Qp(RandomWithDimAndNeq /*tag*/, i32 dim, i32 n_eq)
			: H(dim, dim), g(dim), A(n_eq, dim), b(n_eq), solution(dim + n_eq) {
		A.setRandom();

		// 1/2 (x-sol)T H (x-sol)
		// 1/2 xT H x - (H sol).T x
		solution.setRandom();
		auto primal_solution = solution.topRows(dim);
		auto dual_solution = solution.bottomRows(n_eq);

		{
			H.setRandom();
			H = H * H.transpose();
			H.diagonal().array() += Scalar(1e-3);
		}

		g = -H * primal_solution - A.transpose() * dual_solution;
		b = A * primal_solution;
	}

	auto as_view() -> qp::QpView<Scalar, colmajor, colmajor> {
		return {
				detail::from_eigen_matrix(H),
				detail::from_eigen_vector(g),
				detail::from_eigen_matrix(A),
				detail::from_eigen_vector(b),
				{},
				{},
		};
	}
	auto as_mut() -> qp::QpViewMut<Scalar, colmajor, colmajor> {
		return {
				detail::from_eigen_matrix_mut(H),
				detail::from_eigen_vector_mut(g),
				detail::from_eigen_matrix_mut(A),
				detail::from_eigen_vector_mut(b),
				{},
				{},
		};
	}
};
using Scalar = long double;

DOCTEST_TEST_CASE("qp: random") {
	i32 dim = 30;
	i32 n_eq = 6;

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init.setZero();
	primal_init = -qp.H.llt().solve(qp.g);
	dual_init.setZero();

	Scalar eps_abs = Scalar(1e-10);
	detail::solve_qp( //
			detail::from_eigen_vector_mut(primal_init),
			detail::from_eigen_vector_mut(dual_init),
			qp.as_view(),
			200,
			eps_abs,
			0,
			qp::preconditioner::IdentityPrecond{});

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);
}

DOCTEST_TEST_CASE("qp: start from solution") {
	i32 dim = 30;
	i32 n_eq = 6;

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(dim);
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
	primal_init = qp.solution.topRows(dim);
	dual_init = qp.solution.bottomRows(n_eq);

	Scalar eps_abs = Scalar(1e-10);
	auto iter = detail::solve_qp( //
			detail::from_eigen_vector_mut(primal_init),
			detail::from_eigen_vector_mut(dual_init),
			qp.as_view(),
			200,
			eps_abs,
			0,
			qp::preconditioner::IdentityPrecond{});

	DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);

	DOCTEST_CHECK(iter == 0);
}
