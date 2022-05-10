/**
 * @file wrapper.hpp 
*/

#ifndef PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP
#include <tl/optional.hpp>
#include <qp/Results.hpp>
#include <qp/Settings.hpp>
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
#include <linearsolver/dense/ldlt.hpp>
#include <veg/util/dynstack_alloc.hpp>

auto main() -> int {
	constexpr auto DYN = Eigen::Dynamic;
	using Matrix = Eigen::Matrix<double, DYN, DYN>;
	using Vector = Eigen::Matrix<double, DYN, 1>;
	using Ldlt = linearsolver::dense::Ldlt<double>;
	using veg::dynstack::StackReq;

	// allocate a matrix `a`
	auto a0 = Matrix{
			2,
			2,
	};

	// workspace memory requirements
	auto req =
			Ldlt::factorize_req(2) |          // initial factorization of dim 2
			Ldlt::insert_block_at_req(2, 1) | // or 1 insertion to matrix of dim 2
			Ldlt::delete_at_req(3, 2) |       // or 2 deletions from matrix of dim 3
			Ldlt::solve_in_place_req(1);      // or solve in place with dim 1

	VEG_MAKE_STACK(stack, req);

	Ldlt ldl;

	// fill up the lower triangular part
	// matrix is
	// 1.0 2.0
	// 2.0 3.0
	a0(0, 0) = 1.0;
	a0(1, 0) = 2.0;
	a0(1, 1) = 3.0;

	ldl.factorize(a0, stack);

	// add one column at the index 1
	// matrix is
	// 1.0 4.0 2.0
	// 4.0 5.0 6.0
	// 2.0 6.0 3.0
	auto c = Matrix{3, 1};
	c(0, 0) = 4.0;
	c(1, 0) = 5.0;
	c(2, 0) = 6.0;
	ldl.insert_block_at(1, c, stack);

	// then delete two rows and columns at indices 0 and 2
	// matrix is
	// 5.0
	veg::isize const indices[] = {0, 2};
	ldl.delete_at(indices, 2, stack);

	auto rhs = Vector{1};
	rhs[0] = 5.0;

	ldl.solve_in_place(rhs, stack);
	VEG_ASSERT(rhs[0] == 1.0);
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
				H != tl::nullopt && g != tl::nullopt && A != tl::nullopt &&
				b != tl::nullopt && C != tl::nullopt && u != tl::nullopt &&
				l != tl::nullopt) {
			// if all != tl::nullopt -> initial setup
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
			// some inputs are not equal to tl::nullopt -> do first an update
			data.H = Eigen::
					Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
							H.value());
			data.A = Eigen::
					Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
							A.value());
			data.C = Eigen::
					Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
							C.value());
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
		results.reset_results();
		work.reset_workspace(data.n_in);
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
		}
		{ work.l_scaled = data.l; }
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
	void warm_sart(
			tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		warm_starting(x, y, z, results, settings);
	};
	void cleanup() {
		results.reset_results();
		work.reset_workspace();
	}
};

} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP */
