/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_QP_DENSE_WRAPPER_HPP
#define PROXSUITE_QP_DENSE_WRAPPER_HPP
#include <qp/dense/solver.hpp>
#include <qp/dense/helpers.hpp>
#include <qp/dense/preconditioner/ruiz.hpp>
#include <chrono>

namespace proxsuite {
namespace qp {
namespace dense {
/*!
 * Wrapper class for using proxsuite API with dense backend
 * for solving linearly constrained convex QP with the ProxQp algorithm.  
 * More, precisely, when provided with a QP problem (will its matrices be sparse or dense): 
 * 
 * \f{eqnarray*}{
 * \min_{x} &\frac{1}{2}x^THx& + g^Tx \\
 * Ax &=& b \\
 * l\leq &Cx& \leq u
 * \f}
 * 
 * the solver will provide a global solution \f$ ( x^* , y^* , z^* ) \f$ satisfying the KKT conditions at the defined absolute precision \f$\epsilon_{\text{abs}}\f$:
 * 
 * \f{eqnarray*}{
 * \| Hx^* + g + A^Ty^* + C^Tz^*\| \leq \epsilon_{\text{abs}} \\
 * \| Ax^* -b \| \leq \epsilon_{\text{abs}} \\
 * \| [Cx^* -u]_{+} + [Cx^*-l]_{+} \| \leq \epsilon_{\text{abs}}.
 * \f}
 * 
 * Example usage:
 * ```cpp
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <veg/util/dbg.hpp>
#include <test/include/util.hpp>

using T = double;
auto main() -> int {

	// Generate a random QP problem
	ldlt_test::rand::set_seed(1);
	qp::isize dim = 10;
	qp::isize n_eq(dim / 4);
	qp::isize n_in(dim / 4);
	T strong_convexity_factor(1.e-2);
	T sparsity_factor = 0.15;
	Qp<T> qp{
			random_with_dim_and_neq_and_n_in,
			dim,
			n_eq,
			n_in,
			sparsity_factor,
			strong_convexity_factor}; 
	
	// Solve the problem
	qp::dense::QP<T> Qp{dim, n_eq, n_in}; // creating QP object
	Qp.settings.eps_abs = T(1e-9); // choose accuracy needed
	Qp.setup_dense_matrices(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l); // setup the QP object
	Qp.solve(); // solve the problem

	// Verify solution accuracy
	T pri_res = std::max(
			(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
			(qp::dense::positive_part(qp.C * Qp.results.x - qp.u) +
			qp::dense::negative_part(qp.C * Qp.results.x - qp.l))
					.lpNorm<Eigen::Infinity>());
	T dua_res = (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y +
					qp.C.transpose() * Qp.results.z)
					.lpNorm<Eigen::Infinity>();
	VEG_ASSERT(pri_res <= eps_abs);
	VEG_ASSERT(dua_res <= eps_abs);

	// Some solver statistics
	std::cout << "------solving qp with dim: " << dim
						<< " neq: " << n_eq << " nin: " << n_in << std::endl;
	std::cout << "primal residual: " << pri_res << std::endl;
	std::cout << "dual residual: " << dua_res << std::endl;
	std::cout << "total number of iteration: " << Qp.results.info.iter
						<< std::endl;
}
 * ```
 */
///// QP object
template <typename T>
struct QP {
	Results<T> results;
	Settings<T> settings;
	Model<T> model;
	Workspace<T> work;
	preconditioner::RuizEquilibration<T> ruiz;

	QP(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				model(_dim, _n_eq, _n_in),
				work(_dim, _n_eq, _n_in),
				ruiz(preconditioner::RuizEquilibration<T>{_dim, _n_eq + _n_in}) {
				work.timer.stop();
				}
	void init(
			MatRef<T> H,
			VecRef<T> g,
			MatRef<T> A,
			VecRef<T> b,
			MatRef<T> C,
			VecRef<T> u,
			VecRef<T> l,
			bool compute_preconditioner = true) {
		// dense case
		if (settings.compute_timings){
					work.timer.stop();
					work.timer.start();
		}
		PreconditionerStatus preconditioner_status;
		if (compute_preconditioner){
			preconditioner_status = proxsuite::qp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::qp::PreconditionerStatus::IDENTITY;
		}
		proxsuite::qp::dense::setup(
				H,
				g,
				A,
				b,
				C,
				u,
				l,
				settings,
				model,
				work,
				results,
				ruiz,
				preconditioner_status);

		if (settings.compute_timings){
			results.info.setup_time = work.timer.elapsed().user; // in nanoseconds
		}
	};
	void init(
			const SparseMat<T> H,
			VecRef<T> g,
			const SparseMat<T> A,
			VecRef<T> b,
			const SparseMat<T> C,
			VecRef<T> u,
			VecRef<T> l,
			bool compute_preconditioner = true) {
		// sparse case
		if (settings.compute_timings){
					work.timer.stop();
					work.timer.start();
		}
		PreconditionerStatus preconditioner_status;
		if (compute_preconditioner){
			preconditioner_status = proxsuite::qp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::qp::PreconditionerStatus::IDENTITY;
		}
		proxsuite::qp::dense::setup(
				H,
				g,
				A,
				b,
				C,
				u,
				l,
				settings,
				model,
				work,
				results,
				ruiz,
				preconditioner_status);
		if (settings.compute_timings){
			results.info.setup_time = work.timer.elapsed().user; // in nanoseconds
		}
	};
	void update(
			const tl::optional<MatRef<T>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<MatRef<T>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<MatRef<T>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l,
			bool update_preconditioner = true) {
		// dense case
		if (settings.compute_timings){
					work.timer.stop();
					work.timer.start();
		}
		PreconditionerStatus preconditioner_status;
		if (update_preconditioner){
			preconditioner_status = proxsuite::qp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::qp::PreconditionerStatus::KEEP;
		}
		bool real_update = !(H == tl::nullopt && g == tl::nullopt && A == tl::nullopt &&
		    b == tl::nullopt && C == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt);
		if (real_update) {
			proxsuite::qp::dense::update(H,g,A,b,C,u,l,model);
		}
		proxsuite::qp::dense::setup(
					MatRef<T>(model.H),
					VecRef<T>(model.g),
					MatRef<T>(model.A),
					VecRef<T>(model.b),
					MatRef<T>(model.C),
					VecRef<T>(model.u),
					VecRef<T>(model.l),
					settings,
					model,
					work,
					results,
					ruiz,
					preconditioner_status);
		if (settings.compute_timings){
			results.info.setup_time = work.timer.elapsed().user; // in nanoseconds
		}
	};
	void update(
			const tl::optional<SparseMat<T>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l,
			bool update_preconditioner = true) {
		// sparse case
		if (settings.compute_timings){
					work.timer.stop();
					work.timer.start();
		}
		PreconditionerStatus preconditioner_status;
		if (update_preconditioner){
			preconditioner_status = proxsuite::qp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::qp::PreconditionerStatus::KEEP;
		}
		bool real_update = !(H == tl::nullopt && g == tl::nullopt && A == tl::nullopt &&
		    b == tl::nullopt && C == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt);
		if (real_update) {
			proxsuite::qp::dense::update(H,g,A,b,C,u,l,model);
		}
		proxsuite::qp::dense::setup(
				MatRef<T>(model.H),
				VecRef<T>(model.g),
				MatRef<T>(model.A),
				VecRef<T>(model.b),
				MatRef<T>(model.C),
				VecRef<T>(model.u),
				VecRef<T>(model.l),
				settings,
				model,
				work,
				results,
				ruiz,
				preconditioner_status);
		if (settings.compute_timings){
			results.info.setup_time = work.timer.elapsed().user; // in nanoseconds
		}
	};
	void update(
			const tl::nullopt_t H,
			tl::optional<VecRef<T>> g,
			const tl::nullopt_t A,
			tl::optional<VecRef<T>> b,
			const tl::nullopt_t C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l,
			bool update_preconditioner = true) {
		// treat the case when H, A and C are nullopt, in order to avoid ambiguity through overloading
		if (settings.compute_timings){
					work.timer.stop();
					work.timer.start();
		}
		PreconditionerStatus preconditioner_status;
		if (update_preconditioner){
			preconditioner_status = proxsuite::qp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::qp::PreconditionerStatus::KEEP;
		}
		bool real_update = !(g == tl::nullopt &&
		    b == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt);
		if (real_update) {
			// update the model
			if (g != tl::nullopt) {
				model.g = g.value().eval();
			} 
			if (b != tl::nullopt) {
				model.b = b.value().eval();
			}
			if (u != tl::nullopt) {
				model.u = u.value().eval();
			}
			if (l != tl::nullopt) {
				model.l = l.value().eval();
			} 
		}
		proxsuite::qp::dense::setup(
				MatRef<T>(model.H),
				VecRef<T>(model.g),
				MatRef<T>(model.A),
				VecRef<T>(model.b),
				MatRef<T>(model.C),
				VecRef<T>(model.u),
				VecRef<T>(model.l),
				settings,
				model,
				work,
				results,
				ruiz,
				preconditioner_status);
		if (settings.compute_timings){
			results.info.setup_time = work.timer.elapsed().user; // in nanoseconds
		}
	};
	void solve() {
		qp_solve( //
				settings,
				model,
				results,
				work,
				ruiz);
	};
	void solve(tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		proxsuite::qp::dense::warm_start(x, y, z, results, settings);
		qp_solve( //
				settings,
				model,
				results,
				work,
				ruiz);
	};
	void update_proximal_parameters(
			tl::optional<T> rho, tl::optional<T> mu_eq, tl::optional<T> mu_in) {
		proxsuite::qp::dense::update_proximal_parameters(results, rho, mu_eq, mu_in);
	};
	void cleanup() {
		results.cleanup();
		work.cleanup();
	}
};

template <typename T>
qp::Results<T> solve(const tl::optional<MatRef<T>> H_dense,
			const tl::optional<SparseMat<T>> H_sparse,
			tl::optional<VecRef<T>> g,
			const tl::optional<MatRef<T>> A_dense,
			const tl::optional<SparseMat<T>> A_sparse,
			tl::optional<VecRef<T>> b,
			const tl::optional<MatRef<T>> C_dense,
			const tl::optional<SparseMat<T>> C_sparse,
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
			tl::optional<T> eps_dual_inf
			){
	
	isize n(0);
	isize n_eq(0);
	isize n_in(0);
	if (H_sparse!=tl::nullopt){
		n = H_sparse.value().rows();
		n_eq = A_sparse.value().rows();
		n_in = C_sparse.value().rows();
	}else{
		n = H_dense.value().rows();
		n_eq = A_dense.value().rows();
		n_in = C_dense.value().rows();	
	}
	QP<T> Qp(n, n_eq, n_in);
	if(H_sparse!=tl::nullopt){
		Qp.setup(H_sparse,g,A_sparse,b,C_sparse,u,l); 
	}else{
		Qp.setup(H_dense,g,A_dense,b,C_dense,u,l); 
	}
	
	Qp.update_proximal_parameters(rho,mu_eq,mu_in);
	Qp.warm_start(x,y,z);

	if (eps_abs != tl::nullopt){
		Qp.settings.eps_abs = eps_abs.value();
	}
	if (eps_rel != tl::nullopt){
		Qp.settings.eps_rel = eps_rel.value();
	}
	if (verbose != tl::nullopt){
		Qp.settings.verbose = verbose.value();
	}
	if (alpha_bcl!=tl::nullopt){
		Qp.settings.alpha_bcl = alpha_bcl.value();
	}
	if (beta_bcl != tl::nullopt){
		Qp.settings.beta_bcl = beta_bcl.value();
	}
	if (refactor_dual_feasibility_threshold!=tl::nullopt){
		Qp.settings.refactor_dual_feasibility_threshold = refactor_dual_feasibility_threshold.value();
	}
	if (refactor_rho_threshold!=tl::nullopt){
		Qp.settings.refactor_rho_threshold = refactor_rho_threshold.value();
	}
	if (mu_max_eq!=tl::nullopt){
		Qp.settings.mu_max_eq = mu_max_eq.value();
		Qp.settings.mu_max_eq_inv = T(1)/mu_max_eq.value();
	}
	if (mu_max_in!=tl::nullopt){
		Qp.settings.mu_max_in = mu_max_in.value();
		Qp.settings.mu_max_in_inv = T(1)/mu_max_in.value();
	}
	if (mu_update_factor!=tl::nullopt){
		Qp.settings.mu_update_factor = mu_update_factor.value();
		Qp.settings.mu_update_inv_factor = T(1)/mu_update_factor.value();
	}
	if (cold_reset_mu_eq!=tl::nullopt){
		Qp.settings.cold_reset_mu_eq = cold_reset_mu_eq.value();
		Qp.settings.cold_reset_mu_eq_inv = T(1)/cold_reset_mu_eq.value();
	}
	if (cold_reset_mu_in!=tl::nullopt){
		Qp.settings.cold_reset_mu_in = cold_reset_mu_in.value();
		Qp.settings.cold_reset_mu_in_inv = T(1)/cold_reset_mu_in.value();
	}
	if (max_iter != tl::nullopt){
		Qp.settings.max_iter = max_iter.value();
	}
	if (max_iter_in != tl::nullopt){
		Qp.settings.max_iter_in = max_iter_in.value();
	}
	if (eps_refact != tl::nullopt){
		Qp.settings.eps_refact = eps_refact.value();
	}
	if (nb_iterative_refinement != tl::nullopt){
		Qp.settings.nb_iterative_refinement = nb_iterative_refinement.value();
	}
	if (eps_primal_inf != tl::nullopt){
		Qp.settings.eps_primal_inf = eps_primal_inf.value();
	}
	if (eps_dual_inf != tl::nullopt){
		Qp.settings.eps_dual_inf = eps_dual_inf.value();
	}

	Qp.solve(); 

	return Qp.results;
};


} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_WRAPPER_HPP */
