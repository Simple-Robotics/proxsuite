/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP
#define PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP
#include <tl/optional.hpp>
#include <qp/results.hpp>
#include <qp/settings.hpp>
#include <qp/sparse/solver.hpp>
#include <chrono>

namespace proxsuite {
namespace qp {
namespace sparse {
static constexpr auto DYN = Eigen::Dynamic;
enum { layout = Eigen::RowMajor };
template <typename T,typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

////// SETUP
////// UPDATES ///////

/*!
* Update the proximal parameters of the results object.
*
* @param rho_new primal proximal parameter
* @param mu_eq_new dual equality proximal parameter
* @param mu_in_new dual inequality proximal parameter
* @param results solver result 
*/
template <typename T>
void update_proximal_parameters(
		Results<T>& results,
		tl::optional<T> rho_new,
		tl::optional<T> mu_eq_new,
		tl::optional<T> mu_in_new) {

	if (rho_new != tl::nullopt) {
		results.info.rho = rho_new.value();
	}
	if (mu_eq_new != tl::nullopt) {
		results.info.mu_eq = mu_eq_new.value();
		results.info.mu_eq_inv = T(1) / results.info.mu_eq;
	}
	if (mu_in_new != tl::nullopt) {
		results.info.mu_in = mu_in_new.value();
		results.info.mu_in_inv = T(1) / results.info.mu_in;
	}
}
/*!
* Warm start the results primal and dual variables.
*
* @param x_wm primal proximal parameter
* @param y_wm dual equality proximal parameter
* @param z_wm dual inequality proximal parameter
* @param results solver result 
* @param settings solver settings 
*/
template <typename T>
void warm_starting(
		tl::optional<VecRef<T>> x_wm,
		tl::optional<VecRef<T>> y_wm,
		tl::optional<VecRef<T>> z_wm,
		Results<T>& results,
		Settings<T>& settings) {
	bool real_wm = false;
	if (x_wm != tl::nullopt) {
		results.x = x_wm.value().eval();
		real_wm = true;
	}
	if (y_wm != tl::nullopt) {
		results.y = y_wm.value().eval();
		real_wm = true;
	}
	if (z_wm != tl::nullopt) {
		results.z = z_wm.value().eval();
	}
	if (real_wm) {
		settings.warm_start = true;
	}
}

template <typename T, typename I, typename P>
void qp_setup(
		QpView<T, I> qp,
		Results<T>& results,
		Data<T, I>& data,
		Workspace<T, I>& work,
		P& /*precond*/) {
	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();

	if (results.x.rows() != n) {
		results.x.resize(n);
		results.x.setZero();
	}
	if (results.y.rows() != n_eq) {
		results.y.resize(n_eq);
		results.y.setZero();
	}
	if (results.z.rows() != n_in) {
		results.z.resize(n_in);
		results.z.setZero();
	}

	work._.setup_impl(
			qp, data, P::scale_qp_in_place_req(veg::Tag<T>{}, n, n_eq, n_in));
}
///// QP object
template <typename T,typename I>
struct QP {
public:
	Results<T> results;
	Settings<T> settings;
	Data<T,I> data;
	Workspace<T,I> work;
    preconditioner::RuizEquilibration<T, I> ruiz;

	QP(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				data(),
				work(),ruiz(_dim,_n_eq + _n_in,1e-3,10,preconditioner::Symmetry::UPPER) {}

	void setup_sparse_matrices(
			const tl::optional<SparseMat<T,I>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T,I>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T,I>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l) {
                
        SparseMat<T, I> H_triu = H.value().template triangularView<Eigen::Upper>();
        // only initial setup available (if an update of only one)

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
        
        qp_setup(
                qp,
                results,
                data,
                work,
                ruiz);
	};

	void solve() {

		auto start = std::chrono::high_resolution_clock::now();
		sparse::qp_solve( //
				results,
				data,
				settings,
				work,
                ruiz);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration =
				std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		results.info.solve_time = T(duration.count());
		results.info.run_time = results.info.solve_time + results.info.setup_time;

		if (settings.verbose) {
			std::cout << "------ SOLVER STATISTICS--------" << std::endl;
			std::cout << "iter_ext : " << results.info.iter_ext << std::endl;
			std::cout << "iter : " << results.info.iter << std::endl;
			std::cout << "mu updates : " << results.info.mu_updates << std::endl;
			std::cout << "rho_updates : " << results.info.rho_updates << std::endl;
			std::cout << "objValue : " << results.info.objValue << std::endl;
			std::cout << "solve_time : " << results.info.solve_time << std::endl;
		}
	};

	void update_prox_parameter(
			tl::optional<T> rho, tl::optional<T> mu_eq, tl::optional<T> mu_in) {
		update_proximal_parameters(results, rho, mu_eq, mu_in);
	};
	void warm_start(
			tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		warm_starting(x, y, z, results, settings);
	};
	void cleanup() {
		results.reset_results();
	}
};

template <typename T, typename I>
qp::Results<T> solve(const tl::optional<SparseMat<T,I>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T,I>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T,I>> C,
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

	isize n = H.value().rows();
	isize n_eq = A.value().rows();
	isize n_in = C.value().rows();

	qp::sparse::QP<T,I> Qp(n, n_eq, n_in);
	Qp.setup_sparse_matrices(H,g,A,b,C,u,l); // symbolic factorisation done here

	Qp.update_prox_parameter(rho,mu_eq,mu_in);
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

	Qp.solve(); // numeric facotisation done here

	return Qp.results;
};

} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP */
