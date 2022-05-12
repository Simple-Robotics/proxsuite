/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP
#define PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP
#include <tl/optional.hpp>
#include <qp/Results.hpp>
#include <qp/Settings.hpp>
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
struct QP_sparse {
public:
	Results<T> results;
	Settings<T> settings;
	Data<T,I> data;
	Workspace<T,I> work;
    preconditioner::RuizEquilibration<T, I> ruiz;

	QP_sparse(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				data(),
				work(),ruiz(_dim,_n_eq + _n_in,1e-3,10,preconditioner::Symmetry::UPPER) {}

	void setup_sparse_matrices(
			const tl::optional<SparseMat<T,I>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T,I>> AT,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T,I>> CT,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l) {
  
        // only initial setup available (if an update of only one)

        sparse::QpView<T, I> qp = {
            {linearsolver::sparse::from_eigen, H.value()},
            {linearsolver::sparse::from_eigen, g.value()},
            {linearsolver::sparse::from_eigen, AT.value()},
            {linearsolver::sparse::from_eigen, b.value()},
            {linearsolver::sparse::from_eigen, CT.value()},
            {linearsolver::sparse::from_eigen, l.value()},
            {linearsolver::sparse::from_eigen, u.value()}};
        
        qp_setup(
                qp,
                results,
                data,
                work,
                ruiz);


        /*
		if (H == tl::nullopt && g == tl::nullopt && A == tl::nullopt &&
		    b == tl::nullopt && C == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt) {
			// if all = tl::nullopt -> use previous setup
            isize n = data.dim;
            isize n_eq = data.n_eq;
            isize n_in = data.n_in;
            linearsolver::sparse::MatMut<T, I> kkt = data.kkt_mut();
            auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n);
            linearsolver::sparse::MatMut<T, I> H_ =
                    detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);
            linearsolver::sparse::MatMut<T, I> AT_ =
                    detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, data.A_nnz);
            linearsolver::sparse::MatMut<T, I> CT_ =
                    detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, data.C_nnz);

            sparse::QpView<T, I> qp = {
                H_.as_const(),
                {linearsolver::sparse::from_eigen, g},
                AT_.as_const(),
                {linearsolver::sparse::from_eigen, b},
                CT_.as_const(),
                {linearsolver::sparse::from_eigen, l},
                {linearsolver::sparse::from_eigen, u}};

			qp_setup(
                    qp,
                    results,
                    data,
					work,
                    ruiz);
		} else {
        */


		
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
	void warm_sart(
			tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		warm_starting(x, y, z, results, settings);
	};
	void cleanup() {
		results.reset_results();
	}
};

} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_INCLUDE_QP_SPARSE_WRAPPER_HPP */
