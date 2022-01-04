
#include <ldlt/ldlt.hpp>
#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>

#include <qp/views.hpp>
#include <qp/QPWorkspace.hpp>

#include <qp/proxqp/solver.hpp>
#include <qp/utils.hpp>
#include <qp/precond/ruiz.hpp>
#include <fmt/chrono.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


#include <qp/proxqp/old_solver.hpp>


namespace ldlt {
namespace pybind11 {

template <typename T, Layout L>
using MatRef = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)> const>;
template <typename T, Layout L>
using MatRefMut = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)>>;

template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using VecRefMut = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T, Layout L>
void iterative_solve_with_permut_fact( //
		VecRef<T> rhs,
		VecRefMut<T> sol,
		MatRef<T, L> mat,
		T eps,
		i32 max_it) {
	Ldlt<T> ldl{decompose, mat};
	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);
	auto res = (mat * sol - rhs).eval();
	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;
		res = (mat * sol - rhs);
	}
}

} // namespace pybind11
} // namespace ldlt

namespace qp {
namespace pybind11 {

using namespace ldlt;
template <typename T, Layout L>
using MatRef = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)> const>;
template <typename T, Layout L>
using MatRefMut = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)>>;

template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using VecRefMut = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/*
template <typename T, Layout L>
auto initial_guess_line_search( //
		VecRef<T> x,
		VecRef<T> ye,
		VecRef<T> ze,
		VecRef<T> dw,
		T mu_eq,
		T mu_in,
		T rho,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l) -> T {
	return line_search::initial_guess_line_search(
			{from_eigen, x.eval()},
			{from_eigen, ye.eval()},
			{from_eigen, ze.eval()},
			{from_eigen, dw.eval()},
			mu_eq,
			mu_in,
			rho,
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			});
}

template <typename T, Layout L>
auto correction_guess_line_search( //
		VecRef<T> x,
		VecRef<T> xe,
		VecRef<T> ye,
		VecRef<T> ze,
		VecRef<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l) -> T {
	return line_search::correction_guess_line_search(
			{from_eigen, x.eval()},
			{from_eigen, xe.eval()},
			{from_eigen, ye.eval()},
			{from_eigen, ze.eval()},
			{from_eigen, dx.eval()},
			mu_eq,
			mu_in,
			rho,
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			});
}
*/

template <typename T,Layout L>
void oldQPsolve( //
		VecRefMut<T> x,
		VecRefMut<T> y,
        VecRefMut<T> z,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		isize max_iter,
		isize max_iter_in,
		VecRefMut<T> res_iter,
		T eps_abs,
		T eps_rel){
			
			isize dim = H.eval().rows();
			isize n_eq = A.eval().rows();
			isize n_in = C.eval().rows();
			auto ruiz = qp::preconditioner::RuizEquilibration<T>{
				dim,
				n_eq+n_in,
			};
			qp::detail::QpSolveStats res = qp::detail::old_qpSolve( //
								{from_eigen,x},
								{from_eigen,y},
								{from_eigen,z},
								QpViewBox<T>{
									{from_eigen, H.eval()},
									{from_eigen, g.eval()},
									{from_eigen, A.eval()},
									{from_eigen, b.eval()},
									{from_eigen, C.eval()},
									{from_eigen, u.eval()},
									{from_eigen, l.eval()},
								},
								max_iter,
								max_iter_in,
								eps_abs,
								eps_rel,
								LDLT_FWD(ruiz));
			
			std::cout << "------ SOLVER STATISTICS--------" << std::endl;
			std::cout << "n_ext : " <<  res.n_ext << std::endl;
			std::cout << "n_tot : " <<  res.n_tot << std::endl;
			std::cout << "mu updates : " <<  res.n_mu_updates << std::endl;

			res_iter(0) = res.n_ext;
			res_iter(1) = res.n_tot;
			res_iter(2) = res.n_mu_updates;
}

template <typename T, Layout L>
void QPsetup( //
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		T eps_abs,
		T eps_rel,
		T mu_max_eq,
		T mu_max_in,
		T R,
		T eps_IG,
		T eps_refact,
		isize nb_it_refinement,
		isize max_iter,
		isize max_iter_in,
		const bool VERBOSE) {

	qpsettings._max_iter = max_iter;
	qpsettings._max_iter_in = max_iter_in;
	qpsettings._mu_max_eq = mu_max_eq;
	qpsettings._mu_max_eq_inv = T(1)/mu_max_eq;
	qpsettings._mu_max_in = mu_max_in;
	qpsettings._mu_max_in_inv = T(1)/mu_max_in;
	qpsettings._R = R;
	qpsettings._eps_IG = eps_IG;
	qpsettings._eps_refact = eps_refact;
	qpsettings._eps_abs = eps_abs;
	qpsettings._eps_rel = eps_rel;
	qpsettings._nb_iterative_refinement = nb_it_refinement;

	qpmodel._H = H.eval();
	qpmodel._g = g.eval();
    qpmodel._A = A.eval();
    qpmodel._b = b.eval();
    qpmodel._C = C.eval();
    qpmodel._u = u.eval();
    qpmodel._l = l.eval();

	qpwork._h_scaled = qpmodel._H;
	qpwork._g_scaled = qpmodel._g;
	qpwork._a_scaled = qpmodel._A;
	qpwork._b_scaled = qpmodel._b;
    qpwork._c_scaled = qpmodel._C;
    qpwork._u_scaled = qpmodel._u;
    qpwork._l_scaled = qpmodel._l;

    qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen,qpwork._h_scaled},
			{from_eigen,qpwork._g_scaled},
			{from_eigen,qpwork._a_scaled},
			{from_eigen,qpwork._b_scaled},
			{from_eigen,qpwork._c_scaled},
			{from_eigen,qpwork._u_scaled},
			{from_eigen,qpwork._l_scaled}
	};

	qpwork._ruiz.scale_qp_in_place(qp_scaled,VectorViewMut<T>{from_eigen,qpwork._dw_aug});
    qpwork._dw_aug.setZero();

	qpwork._primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
    qpwork._primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
    qpwork._primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
	qpwork._dual_feasibility_rhs_2 = infty_norm(qpmodel._g);
	qpwork._correction_guess_rhs_g = infty_norm(qpwork._g_scaled);
	
	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qpwork._h_scaled ;
	qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults._rho;	
	qpwork._kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qpwork._a_scaled.transpose();
	qpwork._kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qpwork._a_scaled;
	qpwork._kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
	qpwork._kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults._mu_eq_inv); // mu stores the inverse of mu

	qpwork._ldl.factorize(qpwork._kkt);
	qpwork._rhs.head(qpmodel._dim) = -qpwork._g_scaled;
	qpwork._rhs.segment(qpmodel._dim,qpmodel._n_eq) = qpwork._b_scaled;
	
    qp::detail::iterative_solve_with_permut_fact( //
		qpsettings,
		qpmodel,
		qpresults,
		qpwork,
		T(1),
		qpmodel._dim+qpmodel._n_eq,
        VERBOSE
        );
	
	qpresults._x = qpwork._dw_aug.head(qpmodel._dim);
	qpresults._y = qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq);
	
	qpwork._dw_aug.setZero();

}

/*
template <typename T>
void correction_guess( //
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int) {
			 qp::detail::correction_guess(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				eps_int
			);
}

template <typename T>
void initial_guess_fact( //
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int) {
			 qp::detail::initial_guess_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				eps_int
			);
}

template <typename T>
void iterative_solve_with_permut_fact( //
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int,
		i64 inner_pb_dim) {
			 qp::detail::iterative_solve_with_permut_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				eps_int,
				inner_pb_dim
			);
}
template <typename T>
void BCL_update_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T bcl_eta_ext_init){

		qp::detail::BCL_update_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				primal_feasibility_lhs,
				bcl_eta_ext,
				bcl_eta_in,
				bcl_eta_ext_init);

}

template <typename T>
T global_primal_residual(
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel){
			qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				qpwork,
				qpresults,
				qpmodel
				);
		return primal_feasibility_lhs;
}

template <typename T>
T global_dual_residual(
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults
		){
			qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				qpwork,
				qpresults
			);
		return dual_feasibility_lhs;
};



template <typename T>
void transition_algebra_before_IG_newton(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps_int
	){

	qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._dual_residual_scaled});
	qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._CTz});	

	qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
	qpwork._primal_residual_in_scaled_u -= qpmodel._u;
	qpwork._primal_residual_in_scaled_l -= qpmodel._l;

	qpwork._ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._ze});
	qpwork._primal_residual_in_scaled_u += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u .array() >= 0);
	qpwork._l_active_set_n_l.array() = (qpwork._primal_residual_in_scaled_l.array() <= 0);

	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpwork._primal_residual_in_scaled_u -= qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l -= qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	qpwork._ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u});
	qpwork._ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_l});
	qpwork._ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._ze});
	// rescale value
	isize num_active_inequalities = qpwork._active_inequalities.count();
	isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + num_active_inequalities;


	qp::line_search::active_set_change_new(
			qpwork,
			qpresults,
			qpmodel);

	qpwork._err.head(inner_pb_dim).setZero();
	qpwork._rhs.head(qpmodel._dim).setZero();

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_u(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_l(i);
			}
		} else {
			qpwork._rhs.head(qpmodel._dim).noalias() += qpresults._z(i) * qpwork._c_scaled.row(i); // add CTze_inactif to rhs.head(dim)
		}
	}
	
	qpwork._rhs.head(qpmodel._dim) = -qpwork._dual_residual_scaled; // rhs.head(dim) contains now : -(Hxe + g + ATye + CTze_actif)
	qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq) = -qpwork._primal_residual_eq_scaled;

	{
	detail::iterative_solve_with_permut_fact( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps_int,
			inner_pb_dim);
	}

	qpwork._d_dual_for_eq = qpwork._rhs.head(qpmodel._dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz_actif by definition of the solution

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork._current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork._c_scaled.row(i); 
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork._dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork._c_scaled.row(i);
			}
		}
	}

	// use active_part_z as a temporary variable to permut back dw_aug newton step
	for (isize j = 0; j < qpmodel._n_in; ++j) {
		isize i = qpwork._current_bijection_map(j);
		if (i < qpresults._n_c) {
			//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			//cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;
			
			qpwork._active_part_z(j) = qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i);
			qpwork._Cdx(j) = qpwork._rhs(i + qpmodel._dim + qpmodel._n_eq) + qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + i) * qpresults._mu_in_inv; // mu stores the inverse of mu
			
		} else {
			//dw_aug_(j + dim + n_eq) = -z_(j);
			qpwork._active_part_z(j) = -qpresults._z(j);
			qpwork._Cdx(j) = qpwork._c_scaled.row(j).dot(qpwork._dw_aug.head(qpmodel._dim));
		}
	}
	qpwork._dw_aug.tail(qpmodel._n_in) = qpwork._active_part_z ;

	qpwork._primal_residual_in_scaled_u += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork._ze * qpresults._mu_in_inv; // mu stores the inverse of mu

	//qpwork._d_primal_residual_eq = qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq); // By definition of linear system solution // seems unprecise
	qpwork._d_primal_residual_eq.noalias() = qpwork._a_scaled * qpwork._dw_aug.head(qpmodel._dim);

	qpwork._dual_residual_scaled -= qpwork._CTz; // contains now Hxe+g+ATye
	
}


template <typename T>
T saddle_point(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings){
		
		T res = qp::detail::saddle_point(
			qpwork,
			qpresults,
			qpmodel,
			qpsettings);
		return res;
};

template <typename T>
void newton_step_fact(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T eps) {
			qp::detail::newton_step_fact(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings,
				eps);
};

template <typename T>
void correction_guess_LS(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel) {

			qp::line_search::correction_guess_LS(
								qpwork,
								qpresults,
								qpmodel);
};


template <typename T>
void transition_algebra(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings,
		T primal_feasibility_lhs,
		T bcl_eta_in,
		T err_in){
		
		const bool do_initial_guess_fact = primal_feasibility_lhs < qpsettings._eps_IG || qpmodel._n_in == 0;
		bool do_correction_guess = (!do_initial_guess_fact && qpmodel._n_in != 0) ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in && qpmodel._n_in != 0) ;

		if (do_initial_guess_fact && err_in >= bcl_eta_in ) {

			//
			// ATy contains : Hx_new + rho*(x_new-xe) + ATy_new
			// _primal_residual_eq_scaled contains : Ax_new - b -(y_new-ye)//mu_eq
			// Hence ATy becomes below as wanted : Hx_new + rho*(x_new-xe) + mu_eq * AT(Ax_new-b + ye/mu_eq)
			//
			qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
			qpwork._Hx.noalias() += qpwork._alpha * qpwork._h_scaled * qpwork._dw_aug.head(qpmodel._dim);
			qpwork._ATy.noalias() +=  (qpwork._a_scaled.transpose() * qpwork._primal_residual_eq_scaled) * qpresults._mu_eq ; //mu stores mu
			qpwork._primal_residual_eq_scaled.noalias() += qpresults._y * qpresults._mu_eq_inv ; // contains now Ax_new - b + ye/mu_eq
			qpwork._active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork._dual_residual_scaled.noalias() = qpwork._ATy;
			qpwork._dual_residual_scaled.noalias() +=  qpwork._c_scaled.transpose() * qpwork._active_part_z * qpresults._mu_in ; //mu stores mu  // used for newton step at first iteration

			qpwork._primal_residual_in_scaled_u.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l.noalias() += qpresults._z * qpresults._mu_in_inv; //mu stores the inverse of mu
		}
		if (!do_initial_guess_fact ) {

			qpwork._ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual
			qpwork._primal_residual_eq_scaled.noalias()  += qpwork._ye * qpresults._mu_eq_inv;//mu stores the inverse of mu
			qpwork._ATy.noalias() =  (qpwork._a_scaled.transpose() * qpwork._primal_residual_eq_scaled) * qpresults._mu_eq ; //mu stores mu
			qpwork._ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
			
			qpwork._primal_residual_in_scaled_u.noalias()  += qpwork._ze * qpresults._mu_in_inv;//mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
			qpwork._primal_residual_in_scaled_u -= qpwork._u_scaled;
			qpwork._primal_residual_in_scaled_l -= qpwork._l_scaled;
			qpwork._dual_residual_scaled.noalias() = qpwork._Hx + qpwork._ATy + qpwork._g_scaled;
			qpwork._dual_residual_scaled.noalias() += qpresults._rho * (qpresults._x - qpwork._xe);
			qpwork._active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork._dual_residual_scaled.noalias() +=  qpwork._c_scaled.transpose() * qpwork._active_part_z * qpresults._mu_in ; //mu stores mu  // used for newton step at first iteration

		}

}


template <typename T>
void transition_algebra_before_LS_CG(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpdata<T>& qpmodel){
		
		qpwork._d_dual_for_eq.noalias() = qpwork._h_scaled * qpwork._dw_aug.head(qpmodel._dim);
		//qpwork._d_primal_residual_eq.noalias() = qpwork._dw_aug.segment(qpmodel._dim,qpmodel._n_eq) * qpresults._mu_eq_inv; // by definition Adx = dy / mu : seems unprecise
		qpwork._d_primal_residual_eq.noalias() = qpwork._a_scaled * qpwork._dw_aug.head(qpmodel._dim);
		qpwork._Cdx.noalias() = qpwork._c_scaled * qpwork._dw_aug.head(qpmodel._dim);

}

template <typename T>
T transition_algebra_after_LS_CG(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel){
		
		qpresults._x.noalias() += qpwork._alpha * qpwork._dw_aug.head(qpmodel._dim);

		qpwork._primal_residual_in_scaled_u.noalias() += qpwork._alpha * qpwork._Cdx;
		qpwork._primal_residual_in_scaled_l.noalias() += qpwork._alpha * qpwork._Cdx;
		qpwork._primal_residual_eq_scaled.noalias() += qpwork._alpha * qpwork._d_primal_residual_eq;
		qpresults._y.noalias() = qpwork._primal_residual_eq_scaled * qpresults._mu_eq; //mu stores mu

		qpwork._Hx.noalias() += qpwork._alpha * qpwork._d_dual_for_eq ; // stores Hx
		qpwork._ATy.noalias() = (qpwork._a_scaled).transpose() * qpresults._y ;

		qpresults._z =  (qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l)) *  qpresults._mu_in; //mu stores mu
		T rhs_c = std::max(qpwork._correction_guess_rhs_g, infty_norm( qpwork._Hx));
		rhs_c = std::max(rhs_c, infty_norm(qpwork._ATy));
		qpwork._dual_residual_scaled.noalias() = qpwork._c_scaled.transpose() * qpresults._z ; 
		rhs_c = std::max(rhs_c, infty_norm(qpwork._dual_residual_scaled));
		qpwork._dual_residual_scaled.noalias() += qpwork._Hx + qpwork._g_scaled + qpwork._ATy + qpresults._rho * (qpresults._x - qpwork._xe);
		std::cout << "rhs_c " << rhs_c  << std::endl;
		T err_in = infty_norm(qpwork._dual_residual_scaled);
		return err_in;

}


template <typename T>
void gradient_norm_qpalm(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		T alpha,
		T& gr){

		gr = line_search::gradient_norm_qpalm(
							qpwork,
							qpresults,
							qpmodel,
							alpha);

};

template <typename T>
void initial_guess_LS(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::Qpsettings<T>& qpsettings){
			qp::line_search::initial_guess_LS(
				qpwork,
				qpresults,
				qpmodel,
				qpsettings);
			
};

template <typename T>
void transition_algebra_after_IG_LS(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel
	){

	qpwork._primal_residual_in_scaled_u += (qpwork._alpha * qpwork._Cdx);
	qpwork._primal_residual_in_scaled_l += (qpwork._alpha * qpwork._Cdx);
	qpwork._l_active_set_n_u = (qpwork._primal_residual_in_scaled_u.array() >= 0).matrix();
	qpwork._l_active_set_n_l = (qpwork._primal_residual_in_scaled_l.array() <= 0).matrix();
	qpwork._active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpresults._x.noalias() += qpwork._alpha * qpwork._dw_aug.head(qpmodel._dim);
	qpresults._y.noalias() += qpwork._alpha * qpwork._dw_aug.segment(qpmodel._dim, qpmodel._n_eq);

	
	qpwork._active_part_z = qpresults._z + qpwork._alpha * qpwork._dw_aug.tail(qpmodel._n_in) ;

	qpwork._residual_in_z_u_plus_alpha = (qpwork._active_part_z.array() > 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	qpwork._residual_in_z_l_plus_alpha = (qpwork._active_part_z.array() < 0).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));

	qpresults._z = (qpwork._l_active_set_n_u).select(qpwork._residual_in_z_u_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (qpwork._l_active_set_n_l).select(qpwork._residual_in_z_l_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (!qpwork._l_active_set_n_l.array() && !qpwork._l_active_set_n_u.array()).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) ;
	
	qpwork._primal_residual_eq_scaled.noalias() += qpwork._alpha * qpwork._d_primal_residual_eq;
	qpwork._dual_residual_scaled.noalias() += qpwork._alpha * qpwork._d_dual_for_eq;

	qpwork._ATy = qpwork._dual_residual_scaled ;  // will be used in correction guess if needed : contains Hx_new + rho*(x_new-xe) + g + ATynew

}

template <typename T>
void local_saddle_point(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		T& alpha,
		T& gr) {

			gr = line_search::local_saddle_point(
					qpwork,
					qpresults,
					qpmodel,
					alpha);
};

template <typename T>
void gradient_norm_computation(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		T alpha,
		T& grad_norm) {

		grad_norm = qp::line_search::gradient_norm_computation(
						qpwork,
						qpresults,
						qpmodel,
						alpha);

};
*/

template <typename T,Layout L>
void oldNewQPsolve(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpsettings<T>& qpsettings,
		VecRefMut<T> res_iter){
			
			qp::detail::QpSolveStats res = qp::detail::qpSolve( //
								qpsettings,
								qpmodel,
								qpresults,
								qpwork);

			std::cout << "------ SOLVER STATISTICS--------" << std::endl;
			std::cout << "n_ext : " <<  res.n_ext << std::endl;
			std::cout << "n_tot : " <<  res.n_tot << std::endl;
			std::cout << "mu updates : " <<  res.n_mu_updates << std::endl;

			res_iter(0) = res.n_ext;
			res_iter(1) = res.n_tot;
			res_iter(2) = res.n_mu_updates;
}


} // namespace pybind11
} // namespace qp


PYBIND11_MODULE(inria_ldlt_py, m) {
	m.doc() = R"pbdoc(
INRIA LDLT decomposition
------------------------

  .. currentmodule:: inria_ldlt
  .. autosummary::
     :toctree: _generate

     factorize
  )pbdoc";
	using namespace ldlt;
	using namespace qp;
	// using namespace preconditioner;
	//constexpr auto c = colmajor;
	::pybind11::class_<qp::Qpworkspace<f64>>(m, "Qpworkspace")
        .def(::pybind11::init<i64, i64, i64 &>()) // constructor
        // read-write public data member
        //.def_readwrite("_ruiz", &qp::Qpworkspace<f64>::_ruiz)
		//.def_readonly("_ldl", &qp::Qpworkspace<f64>::_ldl)
		.def_readwrite("_h_scaled", &qp::Qpworkspace<f64>::_h_scaled)
		.def_readwrite("_g_scaled", &qp::Qpworkspace<f64>::_g_scaled)
		.def_readwrite("_a_scaled", &qp::Qpworkspace<f64>::_a_scaled)
		.def_readwrite("_c_scaled", &qp::Qpworkspace<f64>::_c_scaled) 
		.def_readwrite("_b_scaled", &qp::Qpworkspace<f64>::_b_scaled) 
		.def_readwrite("_u_scaled", &qp::Qpworkspace<f64>::_u_scaled) 
		.def_readwrite("_l_scaled", &qp::Qpworkspace<f64>::_l_scaled) 
		.def_readwrite("_xe", &qp::Qpworkspace<f64>::_x_prev) 
		.def_readwrite("_ye", &qp::Qpworkspace<f64>::_y_prev) 
		.def_readwrite("_ze", &qp::Qpworkspace<f64>::_z_prev) 
		.def_readwrite("_kkt", &qp::Qpworkspace<f64>::_kkt) 
		.def_readwrite("_current_bijection_map", &qp::Qpworkspace<f64>::_current_bijection_map)
		.def_readwrite("_new_bijection_map", &qp::Qpworkspace<f64>::_new_bijection_map)
		.def_readwrite("_active_set_up", &qp::Qpworkspace<f64>::_active_set_up)
		.def_readwrite("_active_set_low", &qp::Qpworkspace<f64>::_active_set_low) 
		.def_readwrite("_active_inequalities", &qp::Qpworkspace<f64>::_active_inequalities)
		.def_readwrite("_Hdx", &qp::Qpworkspace<f64>::_Hdx)
		.def_readwrite("_Cdx", &qp::Qpworkspace<f64>::_Cdx)
		.def_readwrite("_Adx", &qp::Qpworkspace<f64>::_Adx)
		.def_readwrite("_active_part_z", &qp::Qpworkspace<f64>::_active_part_z)
		.def_readwrite("_alphas", &qp::Qpworkspace<f64>::_alphas) 
		.def_readwrite("_dw_aug", &qp::Qpworkspace<f64>::_dw_aug) 
		.def_readwrite("_rhs", &qp::Qpworkspace<f64>::_rhs)
		.def_readwrite("_err", &qp::Qpworkspace<f64>::_err) 
		.def_readwrite("_primal_feasibility_rhs_1_eq", &qp::Qpworkspace<f64>::_primal_feasibility_rhs_1_eq)
		.def_readwrite("_primal_feasibility_rhs_1_in_u", &qp::Qpworkspace<f64>::_primal_feasibility_rhs_1_in_u) 
		.def_readwrite("_primal_feasibility_rhs_1_in_l", &qp::Qpworkspace<f64>::_primal_feasibility_rhs_1_in_l) 
		.def_readwrite("_dual_feasibility_rhs_2", &qp::Qpworkspace<f64>::_dual_feasibility_rhs_2)
		.def_readwrite("_correction_guess_rhs_g", &qp::Qpworkspace<f64>::_correction_guess_rhs_g)
		.def_readwrite("_alpha", &qp::Qpworkspace<f64>::_alpha)
		.def_readwrite("_dual_residual_scaled", &qp::Qpworkspace<f64>::_dual_residual_scaled)
		.def_readwrite("_primal_residual_eq_scaled", &qp::Qpworkspace<f64>::_primal_residual_eq_scaled)
		.def_readwrite("_primal_residual_in_scaled_up", &qp::Qpworkspace<f64>::_primal_residual_in_scaled_up)
		.def_readwrite("_primal_residual_in_scaled_low", &qp::Qpworkspace<f64>::_primal_residual_in_scaled_low)
		.def_readwrite("_primal_residual_in_scaled_up_plus_alphaCdx", &qp::Qpworkspace<f64>::_primal_residual_in_scaled_up_plus_alphaCdx)
		.def_readwrite("_primal_residual_in_scaled_low_plus_alphaCdx", &qp::Qpworkspace<f64>::_primal_residual_in_scaled_low_plus_alphaCdx)
		.def_readwrite("_CTz", &qp::Qpworkspace<f64>::_CTz) 
		.def_readwrite("_tmp2", &qp::Qpworkspace<f64>::_tmp2)
		.def_readwrite("_tmp3", &qp::Qpworkspace<f64>::_tmp3);

	::pybind11::class_<qp::Qpresults<f64>>(m, "Qpresults")
        .def(::pybind11::init<i64, i64, i64 &>()) // constructor
        // read-write public data member

		.def_readwrite("_x", &qp::Qpresults<f64>::_x)
		.def_readwrite("_y", &qp::Qpresults<f64>::_y)
		.def_readwrite("_z", &qp::Qpresults<f64>::_z)
		.def_readwrite("_n_c", &qp::Qpresults<f64>::_n_c) 
		.def_readwrite("_mu_eq", &qp::Qpresults<f64>::_mu_eq) 
		.def_readwrite("_mu_eq_inv", &qp::Qpresults<f64>::_mu_eq_inv) 
		.def_readwrite("_mu_in", &qp::Qpresults<f64>::_mu_in) 
		.def_readwrite("_mu_in_inv", &qp::Qpresults<f64>::_mu_in_inv) 
		.def_readwrite("_rho", &qp::Qpresults<f64>::_rho) 
		.def_readwrite("_n_tot", &qp::Qpresults<f64>::_n_tot) 
		.def_readwrite("_n_ext", &qp::Qpresults<f64>::_n_ext) 
		.def_readwrite("_n_mu_change", &qp::Qpresults<f64>::_n_mu_change);

	::pybind11::class_<qp::Qpsettings<f64>>(m, "Qpsettings")
        .def(::pybind11::init()) // constructor
        // read-write public data member

		.def_readwrite("_alpha_bcl", &qp::Qpsettings<f64>::_alpha_bcl)
		.def_readwrite("_beta_bcl", &qp::Qpsettings<f64>::_beta_bcl)
		.def_readwrite("_refactor_dual_feasibility_threshold", &qp::Qpsettings<f64>::_refactor_dual_feasibility_threshold)
		.def_readwrite("_refactor_rho_threshold", &qp::Qpsettings<f64>::_refactor_rho_threshold) 
		.def_readwrite("_refactor_rho_update_factor", &qp::Qpsettings<f64>::_refactor_rho_update_factor) 
		.def_readwrite("_mu_max_eq", &qp::Qpsettings<f64>::_mu_max_eq) 
		.def_readwrite("_mu_max_in", &qp::Qpsettings<f64>::_mu_max_in) 
		.def_readwrite("_mu_max_eq_inv", &qp::Qpsettings<f64>::_mu_max_eq_inv) 
		.def_readwrite("_mu_max_in_inv", &qp::Qpsettings<f64>::_mu_max_in_inv) 
		.def_readwrite("_mu_update_factor", &qp::Qpsettings<f64>::_mu_update_factor)
		.def_readwrite("_mu_update_inv_factor", &qp::Qpsettings<f64>::_mu_update_inv_factor) 

		.def_readwrite("_cold_reset_mu_eq", &qp::Qpsettings<f64>::_cold_reset_mu_eq) 
		.def_readwrite("_cold_reset_mu_in", &qp::Qpsettings<f64>::_cold_reset_mu_in) 
		.def_readwrite("_cold_reset_mu_eq_inv", &qp::Qpsettings<f64>::_cold_reset_mu_eq_inv)

		.def_readwrite("_cold_reset_mu_in_inv", &qp::Qpsettings<f64>::_cold_reset_mu_in_inv) 
		.def_readwrite("_max_iter", &qp::Qpsettings<f64>::_max_iter) 
		.def_readwrite("_max_iter_in", &qp::Qpsettings<f64>::_max_iter_in)

		.def_readwrite("_eps_abs", &qp::Qpsettings<f64>::_eps_abs) 
		.def_readwrite("_eps_rel", &qp::Qpsettings<f64>::_eps_rel) 
		.def_readwrite("_err_IG", &qp::Qpsettings<f64>::_eps_IG)
		.def_readwrite("_R", &qp::Qpsettings<f64>::_R)
		.def_readwrite("_nb_iterative_refinement", &qp::Qpsettings<f64>::_nb_iterative_refinement)
		.def_readwrite("_VERBOSE", &qp::Qpsettings<f64>::_VERBOSE);
	
	::pybind11::class_<qp::Qpdata<f64>>(m, "Qpdata")
        //.def(::pybind11::init()) // constructor
		.def(::pybind11::init<i64, i64, i64 &>()) // constructor
        // read-write public data member

		.def_readonly("_H", &qp::Qpdata<f64>::_H)
		.def_readonly("_g", &qp::Qpdata<f64>::_g)
		.def_readonly("_A", &qp::Qpdata<f64>::_A)
		.def_readonly("_b", &qp::Qpdata<f64>::_b) 
		.def_readonly("_C", &qp::Qpdata<f64>::_C) 
		.def_readonly("_u", &qp::Qpdata<f64>::_u) 
		.def_readonly("_l", &qp::Qpdata<f64>::_l) 
		.def_readonly("_dim", &qp::Qpdata<f64>::_dim) 
		.def_readonly("_n_eq", &qp::Qpdata<f64>::_n_eq) 
		.def_readonly("_n_in", &qp::Qpdata<f64>::_n_in) 
		.def_readonly("_n_total", &qp::Qpdata<f64>::_n_total);



	constexpr auto c = rowmajor;
	/*
 	m.def(
			"initial_guess_line_search",
			&qp::pybind11::initial_guess_line_search<f32, c>);
	m.def(
			"initial_guess_line_search",
			&qp::pybind11::initial_guess_line_search<f64, c>);

 	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f32, c>);
	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f64, c>);

	m.def(
			"correction_guess_line_search",
			&qp::pybind11::correction_guess_line_search<f32, c>);
	m.def(
			"correction_guess_line_search",
			&qp::pybind11::correction_guess_line_search<f64, c>);

	*/
	m.def("oldQPsolve", &qp::pybind11::oldQPsolve<f32, c>);
	m.def("oldQPsolve", &qp::pybind11::oldQPsolve<f64, c>);
	
	m.def("oldNewQPsolve", &qp::pybind11::oldNewQPsolve<f32, c>);
	m.def("oldNewQPsolve", &qp::pybind11::oldNewQPsolve<f64, c>);

	m.def("QPsetup", &qp::pybind11::QPsetup<f32, c>);
	m.def("QPsetup", &qp::pybind11::QPsetup<f64, c>);
	/*
	m.def("correction_guess", &qp::pybind11::correction_guess<f32>);
	m.def("correction_guess", &qp::pybind11::correction_guess<f64>);
	m.def("initial_guess_fact", &qp::pybind11::initial_guess_fact<f32>);
	m.def("initial_guess_fact", &qp::pybind11::initial_guess_fact<f64>);
	m.def("iterative_solve_with_permut_fact", &qp::pybind11::iterative_solve_with_permut_fact<f32>);
	m.def("iterative_solve_with_permut_fact", &qp::pybind11::iterative_solve_with_permut_fact<f64>);
	m.def("BCL_update_fact", &qp::pybind11::BCL_update_fact<f32>);
	m.def("BCL_update_fact", &qp::pybind11::BCL_update_fact<f64>);
	m.def("global_primal_residual", &qp::pybind11::global_primal_residual<f32>);
	m.def("global_primal_residual", &qp::pybind11::global_primal_residual<f64>);
	m.def("global_dual_residual", &qp::pybind11::global_dual_residual<f32>);
	m.def("global_dual_residual", &qp::pybind11::global_dual_residual<f64>);
	m.def("saddle_point", &qp::pybind11::saddle_point<f32>);
	m.def("saddle_point", &qp::pybind11::saddle_point<f64>);
	m.def("newton_step_fact", &qp::pybind11::newton_step_fact<f32>);
	m.def("newton_step_fact", &qp::pybind11::newton_step_fact<f64>);
	m.def("correction_guess_LS", &qp::pybind11::correction_guess_LS<f32>);
	m.def("correction_guess_LS", &qp::pybind11::correction_guess_LS<f64>);
	m.def("gradient_norm_qpalm", &qp::pybind11::gradient_norm_qpalm<f32>);
	m.def("gradient_norm_qpalm", &qp::pybind11::gradient_norm_qpalm<f64>);
	m.def("initial_guess_LS", &qp::pybind11::initial_guess_LS<f32>);
	m.def("initial_guess_LS", &qp::pybind11::initial_guess_LS<f64>);
	m.def("local_saddle_point", &qp::pybind11::local_saddle_point<f32>);
	m.def("local_saddle_point", &qp::pybind11::local_saddle_point<f64>);
	m.def("gradient_norm_computation", &qp::pybind11::gradient_norm_computation<f32>);
	m.def("gradient_norm_computation", &qp::pybind11::gradient_norm_computation<f64>);
	m.def("transition_algebra", &qp::pybind11::transition_algebra<f32>);
	m.def("transition_algebra", &qp::pybind11::transition_algebra<f64>);

	m.def("transition_algebra_before_LS_CG", &qp::pybind11::transition_algebra_before_LS_CG<f32>);
	m.def("transition_algebra_before_LS_CG", &qp::pybind11::transition_algebra_before_LS_CG<f64>);
	m.def("transition_algebra_after_LS_CG", &qp::pybind11::transition_algebra_after_LS_CG<f32>);
	m.def("transition_algebra_after_LS_CG", &qp::pybind11::transition_algebra_after_LS_CG<f64>);
	m.def("transition_algebra_before_IG_newton", &qp::pybind11::transition_algebra_before_IG_newton<f32>);
	m.def("transition_algebra_before_IG_newton", &qp::pybind11::transition_algebra_before_IG_newton<f64>);
	m.def("transition_algebra_after_IG_LS", &qp::pybind11::transition_algebra_after_IG_LS<f32>);
	m.def("transition_algebra_after_IG_LS", &qp::pybind11::transition_algebra_after_IG_LS<f64>);
	*/

	m.attr("__version__") = "dev";
}
