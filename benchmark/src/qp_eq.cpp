#include <ldlt/qp/eq_solver.hpp>
#include <numeric>
#include <util.hpp>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

using Scalar = double;

template <typename T>
void use(T /*unused*/) {}

auto main() -> int {
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using Tag = qp::detail::EqSolverTimer;
	{
		LDLT_GET_DURATIONS("scale qp", Tag, Scalar);
		LDLT_GET_DURATIONS("scale solution", Tag, Scalar);
		LDLT_GET_DURATIONS("set H", Tag, Scalar);
		LDLT_GET_DURATIONS("factorization", Tag, Scalar);
		LDLT_GET_DURATIONS("primal residual", Tag, Scalar);
		LDLT_GET_DURATIONS("mu update", Tag, Scalar);
		LDLT_GET_DURATIONS("dual residual", Tag, Scalar);
		LDLT_GET_DURATIONS("unscale solution", Tag, Scalar);
		LDLT_GET_DURATIONS("newton step", Tag, Scalar);
	}

	using ldlt::i32;
	i32 dim = 112;
	i32 n_eq = 16;
	i32 n_iter = 100;
	Qp<double> qp{random_with_dim_and_n_eq, dim, n_eq};

	Scalar eps_abs = Scalar(1e-10);
	for (i32 i = 0; i < n_iter; ++i) {
		Vec primal_init = -qp.H.llt().solve(qp.g);
		Vec dual_init = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_eq);
		qp::detail::solve_qp( //
				qp::detail::from_eigen_vector_mut(primal_init),
				qp::detail::from_eigen_vector_mut(dual_init),
				qp.as_view(),
				200,
				eps_abs,
				0,
				qp::preconditioner::IdentityPrecond{});
	}

	for (auto c : LDLT_GET_MAP(Tag, Scalar)) {
		using ldlt::detail::Duration;
		auto& durations = c.second.ref;
		auto avg = std::accumulate(durations.begin(), durations.end(), Duration{}) /
		           n_iter;
		fmt::print("{:<20}: {}\n", c.first, avg);
	}
}
