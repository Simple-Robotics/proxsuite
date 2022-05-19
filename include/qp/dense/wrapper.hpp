/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_QP_DENSE_WRAPPER_HPP
#define PROXSUITE_QP_DENSE_WRAPPER_HPP
#include <tl/optional.hpp>
#include <qp/results.hpp>
#include <qp/settings.hpp>
#include <qp/dense/solver.hpp>
#include <chrono>

namespace proxsuite {
namespace qp {
namespace dense {
static constexpr auto DYN = Eigen::Dynamic;
enum { layout = Eigen::RowMajor };
template <typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Mat = Eigen::Matrix<T, DYN, DYN, layout>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

/////// SETUP ////////
/*!
* Setup the linear solver and the parameters x, y and z (through warm starting or default values if warm_start=false in the settings)
*
* @param qpwork solver workspace
* @param qpsettings solver settings
* @param qpmodel solver model
* @param qpresults solver result 
*/
template <typename T>
void initial_guess(
		dense::Workspace<T>& qpwork,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		Results<T>& qpresults) {

	qp::dense::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut,
			qpwork.ldl_stack.as_mut(),
	};
	qpwork.ruiz.scale_qp_in_place(qp_scaled, stack);
	qpwork.dw_aug.setZero();

	qpwork.primal_feasibility_rhs_1_eq = dense::infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = dense::infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = dense::infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = dense::infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = qp::dense::infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			qpresults.info.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-qpresults.info.mu_eq);

	qpwork.ldl.factorize(qpwork.kkt, stack);

	if (!qpsettings.warm_start) {
		qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
		qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;
		iterative_solve_with_permut_fact( //
				qpsettings,
				qpmodel,
				qpresults,
				qpwork,
				T(1),
				qpmodel.dim + qpmodel.n_eq);

		qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
		qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
		qpwork.dw_aug.setZero();
		qpwork.rhs.setZero();
	}
}
/*!
* Setup the QP solver (the linear solver backend being dense).
*
* @param H quadratic cost input defining the QP model
* @param g linear cost input defining the QP model
* @param A equality constraint matrix input defining the QP model
* @param b equality constraint vector input defining the QP model
* @param C inequality constraint matrix input defining the QP model
* @param u lower inequality constraint vector input defining the QP model
* @param l lower inequality constraint vector input defining the QP model
* @param qpwork solver workspace
* @param qpsettings solver settings
* @param qpmodel solver model
* @param qpresults solver result 
*/
template <typename Mat, typename T>
void setup_generic( //
		Mat const& H,
		VecRef<T> g,
		Mat const& A,
		VecRef<T> b,
		Mat const& C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults) {

	auto start = std::chrono::steady_clock::now();
	qpmodel.H = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(H);
	qpmodel.g = g;
	qpmodel.A = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(A);
	qpmodel.b = b;
	qpmodel.C = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(C);
	qpmodel.u = u;
	qpmodel.l = l;

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	initial_guess(qpwork, qpsettings, qpmodel, qpresults);

	auto stop = std::chrono::steady_clock::now();
	auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	qpresults.info.setup_time = T(duration.count());
}

/*!
* Setup the QP solver with dense matrices input (the linear solver backend being dense).
*
* @param H quadratic cost input defining the QP model
* @param g linear cost input defining the QP model
* @param A equality constraint matrix input defining the QP model
* @param b equality constraint vector input defining the QP model
* @param C inequality constraint matrix input defining the QP model
* @param u lower inequality constraint vector input defining the QP model
* @param l lower inequality constraint vector input defining the QP model
* @param qpwork solver workspace
* @param qpsettings solver settings
* @param qpmodel solver model
* @param qpresults solver result 
*/
template <typename T>
void setup_dense( //
		MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults) {
	setup_generic(H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

/*!
* Setup the QP solver with sparse matrices input (the linear solver backend being dense).
*
* @param H quadratic cost input defining the QP model
* @param g linear cost input defining the QP model
* @param A equality constraint matrix input defining the QP model
* @param b equality constraint vector input defining the QP model
* @param C inequality constraint matrix input defining the QP model
* @param u lower inequality constraint vector input defining the QP model
* @param l lower inequality constraint vector input defining the QP model
* @param qpwork solver workspace
* @param qpsettings solver settings
* @param qpmodel solver model
* @param qpresults solver result 
*/
template <typename T>
void setup_sparse( //
		const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults) {
	setup_generic(H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

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
public:
	Results<T> results;
	Settings<T> settings;
	Data<T> data;
	Workspace<T> work;

	QP(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				data(_dim, _n_eq, _n_in),
				work(_dim, _n_eq, _n_in) {}

	void setup_dense_matrices(
			tl::optional<MatRef<T>> H,
			tl::optional<VecRef<T>> g,
			tl::optional<MatRef<T>> A,
			tl::optional<VecRef<T>> b,
			tl::optional<MatRef<T>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l) {

		if (H == tl::nullopt && g == tl::nullopt && A == tl::nullopt &&
		    b == tl::nullopt && C == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt) {
			// if all = tl::nullopt -> use previous setup
			setup_dense(
					MatRef<T>(data.H),
					VecRef<T>(data.g),
					MatRef<T>(data.A),
					VecRef<T>(data.b),
					MatRef<T>(data.C),
					VecRef<T>(data.u),
					VecRef<T>(data.l),
					settings,
					data,
					work,
					results);
		} else if (
				H != tl::nullopt && g != tl::nullopt && A != tl::nullopt &&
				b != tl::nullopt && C != tl::nullopt && u != tl::nullopt &&
				l != tl::nullopt) {
			// if all != tl::nullopt -> initial setup
			setup_dense(
					H.value(),
					g.value(),
					A.value(),
					b.value(),
					C.value(),
					u.value(),
					l.value(),
					settings,
					data,
					work,
					results);
		} else {
			// some input are not equal to tl::nullopt -> do first an update
			update(H, g, A, b, C, u, l);
		}
	};
	void setup_sparse_matrices(
			const tl::optional<SparseMat<T>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l) {

		if (H == tl::nullopt && g == tl::nullopt && A == tl::nullopt &&
		    b == tl::nullopt && C == tl::nullopt && u == tl::nullopt &&
		    l == tl::nullopt) {
			// if all = tl::nullopt -> use previous setup
			setup_generic(
					MatRef<T>(data.H),
					VecRef<T>(data.g),
					MatRef<T>(data.A),
					VecRef<T>(data.b),
					MatRef<T>(data.C),
					VecRef<T>(data.u),
					VecRef<T>(data.l),
					settings,
					data,
					work,
					results);
		} else if (
				(H != tl::nullopt && g != tl::nullopt && A != tl::nullopt &&
				b != tl::nullopt && C != tl::nullopt && u != tl::nullopt &&
				l != tl::nullopt) || (H != tl::nullopt) || (A != tl::nullopt) || (C != tl::nullopt)) {
			// if all != tl::nullopt -> initial setup or re setup as a matrix is involved anyway 
			setup_sparse(
					H.value(),
					g.value(),
					A.value(),
					b.value(),
					C.value(),
					u.value(),
					l.value(),
					settings,
					data,
					work,
					results);
		} else {
			// inputs involved are only vectors -> do an update 

			update(tl::nullopt, g, tl::nullopt, b, tl::nullopt, u, l);
		}
		//setup_sparse(H,g,A,b,C,u,l,settings,data,work,results);
	};

	void solve() {

		auto start = std::chrono::high_resolution_clock::now();
		qp_solve( //
				settings,
				data,
				results,
				work);
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

	void update(
			tl::optional<MatRef<T>> H_,
			tl::optional<VecRef<T>> g_,
			tl::optional<MatRef<T>> A_,
			tl::optional<VecRef<T>> b_,
			tl::optional<MatRef<T>> C_,
			tl::optional<VecRef<T>> u_,
			tl::optional<VecRef<T>> l_) {
		results.cleanup();
		work.cleanup();
		if (g_ != tl::nullopt) {
			data.g = g_.value().eval();
			work.g_scaled = data.g;
		} else {
			work.g_scaled = data.g;
		}
		if (b_ != tl::nullopt) {
			data.b = b_.value().eval();
			work.b_scaled = data.b;
		} else {
			work.b_scaled = data.b;
		}
		if (u_ != tl::nullopt) {
			data.u = u_.value().eval();
			work.u_scaled = data.u;
		} else {
			work.u_scaled = data.u;
		}
		if (l_ != tl::nullopt) {
			data.l = l_.value().eval();
			work.l_scaled = data.l;
		} else { work.l_scaled = data.l; }
		if (H_ != tl::nullopt) {
			if (A_ != tl::nullopt) {
				if (C_ != tl::nullopt) {
					data.H = H_.value().eval();
					data.A = A_.value().eval();
					data.C = C_.value().eval();
				} else {
					//update_matrices(data, work, settings,results, H_, A_, MatrixView<T,rowmajor>{from_eigen,data.C});
					//update_matrices(data, work, settings,results, H_, A_, tl::optional<MatRef<T>>(data.C));
					data.H = H_.value().eval();
					data.A = A_.value().eval();
				}
			} else if (C_ != tl::nullopt) {
				data.H = H_.value().eval();
				data.A = A_.value().eval();
			} else {
				data.H = H_.value().eval();
			}
		} else if (A_ != tl::nullopt) {
			if (C_ != tl::nullopt) {
				data.A = A_.value().eval();
				data.C = C_.value().eval();
			} else {
				data.A = A_.value().eval();
			}
		} else if (C_ != tl::nullopt) {
			data.C = C_.value().eval();
		}
		work.H_scaled = data.H;
		work.C_scaled = data.C;
		work.A_scaled = data.A;

		initial_guess(work, settings, data, results);
	}
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
	qp::dense::QP<T> Qp(n, n_eq, n_in);
	if(H_sparse!=tl::nullopt){
		Qp.setup_sparse_matrices(H_sparse,g,A_sparse,b,C_sparse,u,l); 
	}else{
		Qp.setup_dense_matrices(H_dense,g,A_dense,b,C_dense,u,l); 
	}
	
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

	Qp.solve(); 

	return Qp.results;
};


} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_WRAPPER_HPP */
