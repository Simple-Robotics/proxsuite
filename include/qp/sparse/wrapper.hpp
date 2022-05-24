/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_QP_SPARSE_WRAPPER_HPP
#define PROXSUITE_QP_SPARSE_WRAPPER_HPP
#include <tl/optional.hpp>
#include <qp/results.hpp>
#include <qp/settings.hpp>
#include <qp/sparse/solver.hpp>
#include <qp/sparse/helpers.hpp>

namespace proxsuite {
namespace qp {
namespace sparse {

template <typename T, typename I>
struct QP {
	Results<T> results;
	Settings<T> settings;
	Model<T, I> model;
	Workspace<T, I> work;
	preconditioner::RuizEquilibration<T, I> ruiz;

	QP(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				model(),
				work(),
				ruiz(_dim, _n_eq + _n_in, 1e-3, 10, preconditioner::Symmetry::UPPER) {}

	void init(
			const tl::optional<SparseMat<T, I>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T, I>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T, I>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l,
			bool compute_preconditioner_=true) {

		auto start = std::chrono::steady_clock::now();
		settings.compute_preconditioner = compute_preconditioner_;
		
		SparseMat<T, I> H_triu = H.value().template triangularView<Eigen::Upper>();
		SparseMat<T, I> AT = A.value().transpose();
		SparseMat<T, I> CT = C.value().transpose();
		sparse::QpView<T, I> qp = {
				{linearsolver::sparse::from_eigen, H_triu},
				{linearsolver::sparse::from_eigen, g.value()},
				{linearsolver::sparse::from_eigen, AT},
				{linearsolver::sparse::from_eigen, b.value()},
				{linearsolver::sparse::from_eigen, CT},
				{linearsolver::sparse::from_eigen, l.value()},
				{linearsolver::sparse::from_eigen, u.value()}};
		qp_setup(qp, results, model, work, settings, ruiz, settings.compute_preconditioner);
		auto stop = std::chrono::steady_clock::now();
		auto duration =
				std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		results.info.setup_time = T(duration.count());
	};

	void update(const tl::optional<SparseMat<T, I>> H_,
			tl::optional<VecRef<T>> g_,
			const tl::optional<SparseMat<T, I>> A_,
			tl::optional<VecRef<T>> b_,
			const tl::optional<SparseMat<T, I>> C_,
			tl::optional<VecRef<T>> u_,
			tl::optional<VecRef<T>> l_,
			bool update_preconditioner_ = true){

		settings.update_preconditioner = update_preconditioner_;
		
		isize n = model.dim;
		isize n_eq = model.n_eq;
		isize n_in = model.n_in;
		isize n_tot = n + n_eq + n_in;
		linearsolver::sparse::MatMut<T, I> kkt_unscaled = model.kkt_mut_unscaled();

		auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt_unscaled, n);

		linearsolver::sparse::MatMut<T, I> H_unscaled = 
				detail::middle_cols_mut(kkt_top_n_rows, 0, n, model.H_nnz);

		linearsolver::sparse::MatMut<T, I> AT_unscaled =
				detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, model.A_nnz);

		linearsolver::sparse::MatMut<T, I> CT_unscaled =
				detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, model.C_nnz);

		// update the model
		if (g_ != tl::nullopt) {
			model.g = g_.value();
		} 
		if (b_ != tl::nullopt) {
			model.b = b_.value();
		}
		if (u_ != tl::nullopt) {
			model.u = u_.value();
		}
		if (l_ != tl::nullopt) {
			model.l = l_.value();
		} 
		if (H_ != tl::nullopt) {
			if (A_ != tl::nullopt) {
				if (C_ != tl::nullopt) {
					H_unscaled.to_eigen() = H_.value();
					AT_unscaled.to_eigen() = A_.value().transpose();
					CT_unscaled.to_eigen() = C_.value().transpose();
				} else {
					H_unscaled.to_eigen() = H_.value();
					AT_unscaled.to_eigen() = A_.value().transpose();
				}
			} else if (C_ != tl::nullopt) {
				H_unscaled.to_eigen() = H_.value();
				CT_unscaled.to_eigen() = C_.value().transpose();
			} else {
				H_unscaled.to_eigen() = H_.value();
			}
		} else if (A_ != tl::nullopt) {
			if (C_ != tl::nullopt) {
				AT_unscaled.to_eigen() = A_.value().transpose();
				CT_unscaled.to_eigen() = C_.value().transpose();
			} else {
				AT_unscaled.to_eigen() = A_.value().transpose();
			}
		} else if (C_ != tl::nullopt) {
			CT_unscaled.to_eigen() = C_.value().transpose();
		}

		SparseMat<T, I> H_triu = H_unscaled.to_eigen().template triangularView<Eigen::Upper>();
		sparse::QpView<T, I> qp = {
				{linearsolver::sparse::from_eigen, H_triu},
				{linearsolver::sparse::from_eigen, model.g},
				{linearsolver::sparse::from_eigen, AT_unscaled.to_eigen()},
				{linearsolver::sparse::from_eigen, model.b},
				{linearsolver::sparse::from_eigen, CT_unscaled.to_eigen()},
				{linearsolver::sparse::from_eigen, model.l},
				{linearsolver::sparse::from_eigen, model.u}};
		qp_setup(qp, results, model, work, settings, ruiz, settings.update_preconditioner); // store model value + performs scaling according to chosen options

	};

	void solve() {
		qp_solve( //
				results,
				model,
				settings,
				work,
				ruiz);
	};

	void update_proximal_parameters(
			tl::optional<T> rho, tl::optional<T> mu_eq, tl::optional<T> mu_in) {
		proxsuite::qp::sparse::update_proximal_parameters(results, rho, mu_eq, mu_in);
	};
	void warm_start(
			tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		proxsuite::qp::sparse::warm_start(x, y, z, results, settings);
	};

	void cleanup() { results.cleanup(); }
};

template <typename T, typename I>
qp::Results<T> solve(
		const tl::optional<SparseMat<T, I>> H,
		tl::optional<VecRef<T>> g,
		const tl::optional<SparseMat<T, I>> A,
		tl::optional<VecRef<T>> b,
		const tl::optional<SparseMat<T, I>> C,
		tl::optional<VecRef<T>> u,
		tl::optional<VecRef<T>> l,

		tl::optional<T> eps_abs,
		tl::optional<T> eps_rel,
		tl::optional<T> rho,
		tl::optional<T> mu_eq,
		tl::optional<T> mu_in,
		tl::optional<VecRef<T>> x,
		tl::optional<VecRef<T>> y,
		tl::optional<VecRef<T>> z,
		tl::optional<bool> verbose,
		tl::optional<isize> max_iter,
		tl::optional<T> alpha_bcl,
		tl::optional<T> beta_bcl,
		tl::optional<T> refactor_dual_feasibility_threshold,
		tl::optional<T> refactor_rho_threshold,
		tl::optional<T> mu_max_eq,
		tl::optional<T> mu_max_in,
		tl::optional<T> mu_update_factor,
		tl::optional<T> cold_reset_mu_eq,
		tl::optional<T> cold_reset_mu_in,
		tl::optional<isize> max_iter_in,
		tl::optional<T> eps_refact,
		tl::optional<isize> nb_iterative_refinement,
		tl::optional<T> eps_primal_inf,
		tl::optional<T> eps_dual_inf) {

	isize n = H.value().rows();
	isize n_eq = A.value().rows();
	isize n_in = C.value().rows();

	qp::sparse::QP<T, I> Qp(n, n_eq, n_in);
	Qp.setup(H, g, A, b, C, u, l); // symbolic factorisation done here

	Qp.update_proximal_parameters(rho, mu_eq, mu_in);
	Qp.warm_start(x, y, z);

	if (eps_abs != tl::nullopt) {
		Qp.settings.eps_abs = eps_abs.value();
	}
	if (eps_rel != tl::nullopt) {
		Qp.settings.eps_rel = eps_rel.value();
	}
	if (verbose != tl::nullopt) {
		Qp.settings.verbose = verbose.value();
	}
	if (alpha_bcl != tl::nullopt) {
		Qp.settings.alpha_bcl = alpha_bcl.value();
	}
	if (beta_bcl != tl::nullopt) {
		Qp.settings.beta_bcl = beta_bcl.value();
	}
	if (refactor_dual_feasibility_threshold != tl::nullopt) {
		Qp.settings.refactor_dual_feasibility_threshold =
				refactor_dual_feasibility_threshold.value();
	}
	if (refactor_rho_threshold != tl::nullopt) {
		Qp.settings.refactor_rho_threshold = refactor_rho_threshold.value();
	}
	if (mu_max_eq != tl::nullopt) {
		Qp.settings.mu_max_eq = mu_max_eq.value();
		Qp.settings.mu_max_eq_inv = T(1) / mu_max_eq.value();
	}
	if (mu_max_in != tl::nullopt) {
		Qp.settings.mu_max_in = mu_max_in.value();
		Qp.settings.mu_max_in_inv = T(1) / mu_max_in.value();
	}
	if (mu_update_factor != tl::nullopt) {
		Qp.settings.mu_update_factor = mu_update_factor.value();
		Qp.settings.mu_update_inv_factor = T(1) / mu_update_factor.value();
	}
	if (cold_reset_mu_eq != tl::nullopt) {
		Qp.settings.cold_reset_mu_eq = cold_reset_mu_eq.value();
		Qp.settings.cold_reset_mu_eq_inv = T(1) / cold_reset_mu_eq.value();
	}
	if (cold_reset_mu_in != tl::nullopt) {
		Qp.settings.cold_reset_mu_in = cold_reset_mu_in.value();
		Qp.settings.cold_reset_mu_in_inv = T(1) / cold_reset_mu_in.value();
	}
	if (max_iter != tl::nullopt) {
		Qp.settings.max_iter = max_iter.value();
	}
	if (max_iter_in != tl::nullopt) {
		Qp.settings.max_iter_in = max_iter_in.value();
	}
	if (eps_refact != tl::nullopt) {
		Qp.settings.eps_refact = eps_refact.value();
	}
	if (nb_iterative_refinement != tl::nullopt) {
		Qp.settings.nb_iterative_refinement = nb_iterative_refinement.value();
	}
	if (eps_primal_inf != tl::nullopt) {
		Qp.settings.eps_primal_inf = eps_primal_inf.value();
	}
	if (eps_dual_inf != tl::nullopt) {
		Qp.settings.eps_dual_inf = eps_dual_inf.value();
	}

	Qp.solve(); // numeric facotisation done here

	return Qp.results;
};

} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WRAPPER_HPP */
