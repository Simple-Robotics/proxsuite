#include <qp/eq_solver.hpp>
#include <qp/precond/ruiz.hpp>

#include <numeric>
#include <util.hpp>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

using Scalar = double;

auto main() -> int {
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using ldlt::i32;
	i32 total = 128;
	i32 dim = 66;
	i32 n_eq = total - dim;
	i32 n_iter = 100;
	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};

	Scalar eps_abs = Scalar(1e-10);
	for (i32 i = 0; i < n_iter; ++i) {
		Vec primal_init = Vec::Zero(dim);
		Vec dual_init = Vec::Zero(n_eq);
		auto stats = qp::detail::solve_qp( //
				qp::detail::from_eigen_vector_mut(primal_init),
				qp::detail::from_eigen_vector_mut(dual_init),
				qp.as_view(),
				2000,
				eps_abs,
				0,
				qp::preconditioner::
						RuizEquilibration<Scalar, ldlt::colmajor, ldlt::colmajor>{
								dim, n_eq});
		if (i == 0) {
			fmt::print(
					" - {} iterations, {} mu updates, error: {}\n",
					stats.n_iters,
					stats.n_mu_updates,
					(primal_init - qp.solution.topRows(dim)).norm());
		}
	}

	fmt::print(
			"{:<20} | {:>15} | {:>15} | head..tail\n",
			"",
			"total avg",
			"section avg");

	for (auto c : LDLT_GET_MAP(Scalar)["ruiz equilibration"]) {
		using ldlt::detail::Duration;
		auto& durations = c.second.ref;
		auto avg = std::accumulate(durations.begin(), durations.end(), Duration{}) /
		           n_iter;
		auto section_avg =
				std::accumulate(durations.begin(), durations.end(), Duration{}) /
				durations.size();

		fmt::print("{:<20} | {:>15} | {:>15} | ", c.first, avg, section_avg);

		if (durations.size() <= 6) {
			fmt::print("{}\n", durations);
		} else {
			fmt::print( //
					"[{}, {}, {}, ..., {}, {}, {}]\n",
					durations[0],
					durations[1],
					durations[2],
					durations[durations.size() - 1],
					durations[durations.size() - 2],
					durations[durations.size() - 3]);
		}
	}
}
