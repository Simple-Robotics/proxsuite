#include <ldlt/ldlt.hpp>
#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>

#include <qp/views.hpp>
#include <qp/QPWorkspace.hpp>

#include <qp/proxqp/solver.hpp>
#include <qp/proxqp/solver_eq.hpp>
#include <qp/utils.hpp>
#include <qp/precond/ruiz.hpp>
#include <fmt/chrono.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

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

template <typename T, Layout L>
void QPsetup( //
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,

		T eps_abs = 1.e-9,
		T eps_rel = 0,
		const bool VERBOSE = true

) {

	QPSettings.eps_abs = eps_abs;
	QPSettings.eps_rel = eps_rel;
	QPSettings.verbose = VERBOSE;

	qpmodel.H = H.eval();
	qpmodel.g = g.eval();
	qpmodel.A = A.eval();
	qpmodel.b = b.eval();
	qpmodel.C = C.eval();
	qpmodel.u = u.eval();
	qpmodel.l = l.eval();

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	qpwork.ruiz.scale_qp_in_place(
			qp_scaled, VectorViewMut<T>{from_eigen, qpwork.dw_aug});
	qpwork.dw_aug.setZero();

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv); // mu stores the inverse of mu

{
    LDLT_MAKE_STACK(stack, ldlt::Ldlt<T>::factor_req(qpwork.kkt.rows()));
    qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));
  }
	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	qp::detail::iterative_solve_with_permut_fact( //
			QPSettings,
			qpmodel,
			QPResults,
			qpwork,
			T(1),
			qpmodel.dim + qpmodel.n_eq);

	QPResults.x = qpwork.dw_aug.head(qpmodel.dim);
	QPResults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

	qpwork.dw_aug.setZero();
}
*/



template <typename T, Layout L>
void QPupdateMatrice( //
		MatRef<T, L> H,
		MatRef<T, L> A,
		MatRef<T, L> C,
		const qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,
		const bool update = false) {

	// update matrices

	qpmodel.H = H.eval();
	qpmodel.A = A.eval();
	qpmodel.C = C.eval();

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	qpwork.ruiz.scale_qp_in_place(
			qp_scaled, VectorViewMut<T>{from_eigen, qpwork.dw_aug});
	qpwork.dw_aug.setZero();

	// re update all other variables
	if (update) {
		QPResults.reset_results();
	}

	// perform warm start

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv); // mu stores the inverse of mu

  {
    LDLT_MAKE_STACK(stack, ldlt::Ldlt<T>::factor_req(qpwork.kkt.rows()));
    qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));
  }
	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	qp::detail::iterative_solve_with_permut_fact( //
			QPSettings,
			qpmodel,
			QPResults,
			qpwork,
			T(1),
			qpmodel.dim + qpmodel.n_eq);

	QPResults.x = qpwork.dw_aug.head(qpmodel.dim);
	QPResults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

	qpwork.dw_aug.setZero();
}

template <typename T>
void QPupdateVectors( //
		VecRef<T> g,
		VecRef<T> b,
		VecRef<T> u,
		VecRef<T> l,
		const qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,
		const bool update = false) {

	// update vectors

	qpmodel.g = g.eval();
	qpmodel.b = b.eval();
	qpmodel.u = u.eval();
	qpmodel.l = l.eval();

	qpwork.g_scaled = qpmodel.g;
	qpwork.b_scaled = qpmodel.b;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qpwork.ruiz.scale_primal_in_place(
			VectorViewMut<T>{from_eigen, qpwork.g_scaled});
	qpwork._scale_dual_in_place_eq(VectorViewMut<T>{from_eigen, qpwork.b_scaled});
	qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.u_scaled});
	qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.l_scaled});

	// re update all other variables --> no need for warm start (reuse previous
	// one ?) ? to finish
}

/*
template <typename T>
void correction_guess( //
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T eps_int) {
       qp::detail::correction_guess(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings,
        eps_int
      );
}

template <typename T>
void initial_guess_fact( //
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T eps_int) {
       qp::detail::initial_guess_fact(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings,
        eps_int
      );
}

template <typename T>
void iterative_solve_with_permut_fact( //
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T eps_int,
    i64 inner_pb_dim) {
       qp::detail::iterative_solve_with_permut_fact(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings,
        eps_int,
        inner_pb_dim
      );
}
template <typename T>
void BCL_update_fact(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T primal_feasibility_lhs,
    T& bcl_eta_ext,
    T& bcl_eta_in,
    T bcl_eta_ext_init){

    qp::detail::BCL_update_fact(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings,
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
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel){
      qp::detail::global_primal_residual(
        primal_feasibility_lhs,
        primal_feasibility_eq_rhs_0,
        primal_feasibility_in_rhs_0,
        qpwork,
        QPResults,
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
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults
    ){
      qp::detail::global_dual_residual(
        dual_feasibility_lhs,
        dual_feasibility_rhs_0,
        dual_feasibility_rhs_1,
        dual_feasibility_rhs_3,
        qpwork,
        QPResults
      );
    return dual_feasibility_lhs;
};



template <typename T>
void transition_algebra_before_IG_newton(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T eps_int
  ){

  qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork.dual_residual_scaled});
  qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork.CTz});

  qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
  qpwork._primal_residual_in_scaled_u -= qpmodel.u;
  qpwork._primal_residual_in_scaled_l -= qpmodel.l;

  qpwork.ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.ze});
  qpwork._primal_residual_in_scaled_u += qpwork.ze * QPResults.mu_in_inv; // mu
stores the inverse of mu qpwork._primal_residual_in_scaled_l += qpwork.ze *
QPResults.mu_in_inv; // mu stores the inverse of mu

  qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u
.array() >= 0); qpwork._l_active_set_n_l.array() =
(qpwork._primal_residual_in_scaled_l.array() <= 0);

  qpwork.active_inequalities = qpwork._l_active_set_n_u ||
qpwork._l_active_set_n_l;

  qpwork._primal_residual_in_scaled_u -= qpwork.ze * QPResults.mu_in_inv; // mu
stores the inverse of mu qpwork._primal_residual_in_scaled_l -= qpwork.ze *
QPResults.mu_in_inv; // mu stores the inverse of mu

  qpwork.ruiz.scale_primal_residual_in_place_in(
      VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u});
  qpwork.ruiz.scale_primal_residual_in_place_in(
      VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_l});
  qpwork.ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.ze});
  // rescale value
  isize numactive_inequalities = qpwork.active_inequalities.count();
  isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;


  qp::line_search::active_set_change_new(
      qpwork,
      QPResults,
      qpmodel);

  qpwork.err.head(inner_pb_dim).setZero();
  qpwork.rhs.head(qpmodel.dim).setZero();

  for (isize i = 0; i < qpmodel.n_in; i++) {
    isize j = qpwork.current_bijection_map(i);
    if (j < QPResults._n_c) {
      if (qpwork._l_active_set_n_u(i)) {
        qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
-qpwork._primal_residual_in_scaled_u(i); } else if (qpwork._l_active_set_n_l(i))
{ qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
-qpwork._primal_residual_in_scaled_l(i);
      }
    } else {
      qpwork.rhs.head(qpmodel.dim).noalias() += QPResults.z(i) *
qpwork.C_scaled.row(i); // add CTze_inactif to rhs.head(dim)
    }
  }

  qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled; // rhs.head(dim)
contains now : -(Hxe + g + ATye + CTze_actif) qpwork.rhs.segment(qpmodel.dim,
qpmodel.n_eq) = -qpwork.primal_residual_eq_scaled;

  {
  detail::iterative_solve_with_permut_fact( //
      qpwork,
      QPResults,
      qpmodel,
      QPSettings,
      eps_int,
      inner_pb_dim);
  }

  qpwork._d_dual_for_eq = qpwork.rhs.head(qpmodel.dim); // d_dual_for_eq_ =
-dual_for_eq_ -C^T dz_actif by definition of the solution

  for (isize i = 0; i < qpmodel.n_in; i++) {
    isize j = qpwork.current_bijection_map(i);
    if (j < QPResults._n_c) {
      if (qpwork._l_active_set_n_u(i)) {
        qpwork._d_dual_for_eq.noalias() -= qpwork.dw_aug(j + qpmodel.dim +
qpmodel.n_eq) * qpwork.C_scaled.row(i); } else if (qpwork._l_active_set_n_l(i))
{ qpwork._d_dual_for_eq.noalias() -= qpwork.dw_aug(j + qpmodel.dim +
qpmodel.n_eq) * qpwork.C_scaled.row(i);
      }
    }
  }

  // use active_part_z as a temporary variable to permut back dw_aug newton step
  for (isize j = 0; j < qpmodel.n_in; ++j) {
    isize i = qpwork.current_bijection_map(j);
    if (i < QPResults._n_c) {
      //dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
      //cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;

      qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
      qpwork.Cdx(j) = qpwork.rhs(i + qpmodel.dim + qpmodel.n_eq) +
qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i) * QPResults.mu_in_inv; // mu
stores the inverse of mu

    } else {
      //dw_aug_(j + dim + n_eq) = -z_(j);
      qpwork.active_part_z(j) = -QPResults.z(j);
      qpwork.Cdx(j) =
qpwork.C_scaled.row(j).dot(qpwork.dw_aug.head(qpmodel.dim));
    }
  }
  qpwork.dw_aug.tail(qpmodel.n_in) = qpwork.active_part_z ;

  qpwork._primal_residual_in_scaled_u += qpwork.ze * QPResults.mu_in_inv; // mu
stores the inverse of mu qpwork._primal_residual_in_scaled_l += qpwork.ze *
QPResults.mu_in_inv; // mu stores the inverse of mu

  //qpwork._d_primal_residual_eq = qpwork.rhs.segment(qpmodel.dim,
qpmodel.n_eq); // By definition of linear system solution // seems unprecise
  qpwork._d_primal_residual_eq.noalias() = qpwork.A_scaled *
qpwork.dw_aug.head(qpmodel.dim);

  qpwork.dual_residual_scaled -= qpwork.CTz; // contains now Hxe+g+ATye

}


template <typename T>
T saddle_point(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings){

    T res = qp::detail::saddle_point(
      qpwork,
      QPResults,
      qpmodel,
      QPSettings);
    return res;
};

template <typename T>
void newton_step_fact(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T eps) {
      qp::detail::newton_step_fact(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings,
        eps);
};

template <typename T>
void correction_guess_LS(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel) {

      qp::line_search::correction_guess_LS(
                qpwork,
                QPResults,
                qpmodel);
};


template <typename T>
void transition_algebra(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings,
    T primal_feasibility_lhs,
    T bcl_eta_in,
    T err_in){

    const bool do_initial_guess_fact = primal_feasibility_lhs <
QPSettings.eps_IG || qpmodel.n_in == 0; bool do_correction_guess =
(!do_initial_guess_fact && qpmodel.n_in != 0) || (do_initial_guess_fact &&
err_in >= bcl_eta_in && qpmodel.n_in != 0) ;

    if (do_initial_guess_fact && err_in >= bcl_eta_in ) {

      //
      // ATy contains : Hx_new + rho*(x_new-xe) + ATy_new
      // primal_residual_eq_scaled contains : Ax_new - b -(y_new-ye)//mu_eq
      // Hence ATy becomes below as wanted : Hx_new + rho*(x_new-xe) + mu_eq *
AT(Ax_new-b + ye/mu_eq)
      //
      qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
      qpwork._Hx.noalias() += qpwork.alpha * qpwork.H_scaled *
qpwork.dw_aug.head(qpmodel.dim); qpwork._ATy.noalias() +=
(qpwork.A_scaled.transpose() * qpwork.primal_residual_eq_scaled) *
QPResults.mu_eq ; //mu stores mu qpwork.primal_residual_eq_scaled.noalias() +=
QPResults.y * QPResults.mu_eq_inv ; // contains now Ax_new - b + ye/mu_eq
      qpwork.active_part_z =
qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) +
qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
      qpwork.dual_residual_scaled.noalias() = qpwork._ATy;
      qpwork.dual_residual_scaled.noalias() +=  qpwork.C_scaled.transpose() *
qpwork.active_part_z * QPResults.mu_in ; //mu stores mu  // used for newton step
at first iteration

      qpwork._primal_residual_in_scaled_u.noalias() += QPResults.z *
QPResults.mu_in_inv; //mu stores the inverse of mu
      qpwork._primal_residual_in_scaled_l.noalias() += QPResults.z *
QPResults.mu_in_inv; //mu stores the inverse of mu
    }
    if (!do_initial_guess_fact ) {

      qpwork.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen,
qpwork._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains
Cx unscaled from global primal residual
      qpwork.primal_residual_eq_scaled.noalias()  += qpwork.ye *
QPResults.mu_eq_inv;//mu stores the inverse of mu qpwork._ATy.noalias() =
(qpwork.A_scaled.transpose() * qpwork.primal_residual_eq_scaled) *
QPResults.mu_eq ; //mu stores mu
      qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});

      qpwork._primal_residual_in_scaled_u.noalias()  += qpwork.ze *
QPResults.mu_in_inv;//mu stores the inverse of mu
      qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
      qpwork._primal_residual_in_scaled_u -= qpwork.u_scaled;
      qpwork._primal_residual_in_scaled_l -= qpwork.l_scaled;
      qpwork.dual_residual_scaled.noalias() = qpwork._Hx + qpwork._ATy +
qpwork.g_scaled; qpwork.dual_residual_scaled.noalias() += QPResults._rho *
(QPResults.x - qpwork.xe); qpwork.active_part_z =
qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) +
qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
      qpwork.dual_residual_scaled.noalias() +=  qpwork.C_scaled.transpose() *
qpwork.active_part_z * QPResults.mu_in ; //mu stores mu  // used for newton step
at first iteration

    }

}


template <typename T>
void transition_algebra_before_LS_CG(
    qp::QPWorkspace<T>& qpwork,
    qp::QPData<T>& qpmodel){

    qpwork._d_dual_for_eq.noalias() = qpwork.H_scaled *
qpwork.dw_aug.head(qpmodel.dim);
    //qpwork._d_primal_residual_eq.noalias() =
qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq) * QPResults.mu_eq_inv; // by
definition Adx = dy / mu : seems unprecise
    qpwork._d_primal_residual_eq.noalias() = qpwork.A_scaled *
qpwork.dw_aug.head(qpmodel.dim); qpwork.Cdx.noalias() = qpwork.C_scaled *
qpwork.dw_aug.head(qpmodel.dim);

}

template <typename T>
T transition_algebra_after_LS_CG(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel){

    QPResults.x.noalias() += qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim);

    qpwork._primal_residual_in_scaled_u.noalias() += qpwork.alpha * qpwork.Cdx;
    qpwork._primal_residual_in_scaled_l.noalias() += qpwork.alpha * qpwork.Cdx;
    qpwork.primal_residual_eq_scaled.noalias() += qpwork.alpha *
qpwork._d_primal_residual_eq; QPResults.y.noalias() =
qpwork.primal_residual_eq_scaled * QPResults.mu_eq; //mu stores mu

    qpwork._Hx.noalias() += qpwork.alpha * qpwork._d_dual_for_eq ; // stores Hx
    qpwork._ATy.noalias() = (qpwork.A_scaled).transpose() * QPResults.y ;

    QPResults.z =
(qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) +
qp::detail::negative_part(qpwork._primal_residual_in_scaled_l)) *
QPResults.mu_in; //mu stores mu T rhs_c =
std::max(qpwork.correction_guess_rhs_g, infty_norm( qpwork._Hx)); rhs_c =
std::max(rhs_c, infty_norm(qpwork._ATy)); qpwork.dual_residual_scaled.noalias()
= qpwork.C_scaled.transpose() * QPResults.z ; rhs_c = std::max(rhs_c,
infty_norm(qpwork.dual_residual_scaled)); qpwork.dual_residual_scaled.noalias()
+= qpwork._Hx + qpwork.g_scaled + qpwork._ATy + QPResults._rho * (QPResults.x -
qpwork.xe); std::cout << "rhs_c " << rhs_c  << std::endl; T err_in =
infty_norm(qpwork.dual_residual_scaled); return err_in;

}


template <typename T>
void gradient_norm_qpalm(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    T alpha,
    T& gr){

    gr = line_search::gradient_norm_qpalm(
              qpwork,
              QPResults,
              qpmodel,
              alpha);

};

template <typename T>
void initial_guess_LS(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    qp::QPSettings<T>& QPSettings){
      qp::line_search::initial_guess_LS(
        qpwork,
        QPResults,
        qpmodel,
        QPSettings);

};

template <typename T>
void transition_algebra_after_IG_LS(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel
  ){

  qpwork._primal_residual_in_scaled_u += (qpwork.alpha * qpwork.Cdx);
  qpwork._primal_residual_in_scaled_l += (qpwork.alpha * qpwork.Cdx);
  qpwork._l_active_set_n_u = (qpwork._primal_residual_in_scaled_u.array() >=
0).matrix(); qpwork._l_active_set_n_l =
(qpwork._primal_residual_in_scaled_l.array() <= 0).matrix();
  qpwork.active_inequalities = qpwork._l_active_set_n_u ||
qpwork._l_active_set_n_l;

  QPResults.x.noalias() += qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim);
  QPResults.y.noalias() += qpwork.alpha * qpwork.dw_aug.segment(qpmodel.dim,
qpmodel.n_eq);


  qpwork.active_part_z = QPResults.z + qpwork.alpha *
qpwork.dw_aug.tail(qpmodel.n_in) ;

  qpwork._residual_in_z_u_plus_alpha = (qpwork.active_part_z.array() >
0).select(qpwork.active_part_z,
Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in));
  qpwork._residual_in_z_l_plus_alpha = (qpwork.active_part_z.array() <
0).select(qpwork.active_part_z,
Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in));

  QPResults.z =
(qpwork._l_active_set_n_u).select(qpwork._residual_in_z_u_plus_alpha,
Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in)) +
           (qpwork._l_active_set_n_l).select(qpwork._residual_in_z_l_plus_alpha,
Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in)) +
           (!qpwork._l_active_set_n_l.array() &&
!qpwork._l_active_set_n_u.array()).select(qpwork.active_part_z,
Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in)) ;

  qpwork.primal_residual_eq_scaled.noalias() += qpwork.alpha *
qpwork._d_primal_residual_eq; qpwork.dual_residual_scaled.noalias() +=
qpwork.alpha * qpwork._d_dual_for_eq;

  qpwork._ATy = qpwork.dual_residual_scaled ;  // will be used in correction
guess if needed : contains Hx_new + rho*(x_new-xe) + g + ATynew

}

template <typename T>
void local_saddle_point(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    T& alpha,
    T& gr) {

      gr = line_search::local_saddle_point(
          qpwork,
          QPResults,
          qpmodel,
          alpha);
};

template <typename T>
void gradient_norm_computation(
    qp::QPWorkspace<T>& qpwork,
    qp::QPResults<T>& QPResults,
    qp::QPData<T>& qpmodel,
    T alpha,
    T& grad_norm) {

    grad_norm = qp::line_search::gradient_norm_computation(
            qpwork,
            QPResults,
            qpmodel,
            alpha);

};
*/

template <typename T, Layout L>
void QPsolve(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& QPResults,
		qp::QPWorkspace<T>& qpwork,
		const qp::QPSettings<T>& qpsettings) {

	auto start = std::chrono::high_resolution_clock::now();
	qp::detail::qp_solve( //
			qpsettings,
			qpmodel,
			QPResults,
			qpwork);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	QPResults.timing = duration.count();

	if (qpsettings.verbose) {
		std::cout << "------ SOLVER STATISTICS--------" << std::endl;
		std::cout << "n_ext : " << QPResults.n_ext << std::endl;
		std::cout << "n_tot : " << QPResults.n_tot << std::endl;
		std::cout << "mu updates : " << QPResults.n_mu_change << std::endl;
		std::cout << "objValue : " << QPResults.objValue << std::endl;
		std::cout << "timing : " << QPResults.timing << std::endl;
	}
}

template <typename T>
void QPreset(
		const qp::QPData<T>& qpmodel,
		const qp::QPSettings<T>& qpsettings,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	qpwork.kkt.diagonal().head(qpmodel.dim).array() -= qpresults.rho;
	qpresults.reset_results();
	qpwork.reset_results(qpmodel.n_in);

	qpwork.kkt.diagonal().head(qpmodel.dim).array() += qpresults.rho;
	qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
			-qpresults.mu_eq_inv;
{
    LDLT_MAKE_STACK(stack, ldlt::Ldlt<T>::factor_req(qpwork.kkt.rows()));
    qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));
  }

	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	qp::detail::iterative_solve_with_permut_fact( //
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

template <typename T, Layout L>
void QPsolveEq(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		const qp::QPSettings<T>& qpsettings) {

	auto start = std::chrono::high_resolution_clock::now();
	qp::detail::qp_solve_eq( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "------ SOLVER STATISTICS--------" << std::endl;
	std::cout << "n_ext : " << qpresults.n_ext << std::endl;
	std::cout << "n_tot : " << qpresults.n_tot << std::endl;
	std::cout << "mu updates : " << qpresults.n_mu_change << std::endl;
	std::cout << "objValue : " << qpresults.objValue << std::endl;
	qpresults.timing = duration.count();
	std::cout << "timing : " << qpresults.timing << std::endl;
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
	::pybind11::class_<qp::QPWorkspace<f64>>(m, "QPWorkspace")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
																							 // read-write public data member
	    //.def_readwrite("ruiz", &qp::QPWorkspace<f64>::ruiz)
			//.def_readonly("ldl", &qp::QPWorkspace<f64>::ldl)
			.def_readwrite("H_scaled", &qp::QPWorkspace<f64>::H_scaled)
			.def_readwrite("g_scaled", &qp::QPWorkspace<f64>::g_scaled)
			.def_readwrite("A_scaled", &qp::QPWorkspace<f64>::A_scaled)
			.def_readwrite("C_scaled", &qp::QPWorkspace<f64>::C_scaled)
			.def_readwrite("b_scaled", &qp::QPWorkspace<f64>::b_scaled)
			.def_readwrite("u_scaled", &qp::QPWorkspace<f64>::u_scaled)
			.def_readwrite("l_scaled", &qp::QPWorkspace<f64>::l_scaled)
			.def_readwrite("x_prev", &qp::QPWorkspace<f64>::x_prev)
			.def_readwrite("y_prev", &qp::QPWorkspace<f64>::y_prev)
			.def_readwrite("z_prev", &qp::QPWorkspace<f64>::z_prev)
			.def_readwrite("kkt", &qp::QPWorkspace<f64>::kkt)
			.def_readwrite(
					"current_bijection_map", &qp::QPWorkspace<f64>::current_bijection_map)
			.def_readwrite(
					"new_bijection_map", &qp::QPWorkspace<f64>::new_bijection_map)
			.def_readwrite("active_set_up", &qp::QPWorkspace<f64>::active_set_up)
			.def_readwrite("active_set_low", &qp::QPWorkspace<f64>::active_set_low)
			.def_readwrite(
					"active_inequalities", &qp::QPWorkspace<f64>::active_inequalities)
			.def_readwrite("Hdx", &qp::QPWorkspace<f64>::Hdx)
			.def_readwrite("Cdx", &qp::QPWorkspace<f64>::Cdx)
			.def_readwrite("Adx", &qp::QPWorkspace<f64>::Adx)
			.def_readwrite("active_part_z", &qp::QPWorkspace<f64>::active_part_z)
			.def_readwrite("alphas", &qp::QPWorkspace<f64>::alphas)
			.def_readwrite("dw_aug", &qp::QPWorkspace<f64>::dw_aug)
			.def_readwrite("rhs", &qp::QPWorkspace<f64>::rhs)
			.def_readwrite("err", &qp::QPWorkspace<f64>::err)
			.def_readwrite(
					"primal_feasibility_rhs_1_eq",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_eq)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_u",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_in_u)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_l",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_in_l)
			.def_readwrite(
					"dual_feasibility_rhs_2",
					&qp::QPWorkspace<f64>::dual_feasibility_rhs_2)
			.def_readwrite(
					"correction_guess_rhs_g",
					&qp::QPWorkspace<f64>::correction_guess_rhs_g)
			.def_readwrite("alpha", &qp::QPWorkspace<f64>::alpha)
			.def_readwrite(
					"dual_residual_scaled", &qp::QPWorkspace<f64>::dual_residual_scaled)
			.def_readwrite(
					"primal_residual_eq_scaled",
					&qp::QPWorkspace<f64>::primal_residual_eq_scaled)
			.def_readwrite(
					"primal_residual_in_scaled_up",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_up)
			.def_readwrite(
					"primal_residual_in_scaled_low",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_low)
			.def_readwrite(
					"primal_residual_in_scaled_up_plus_alphaCdx",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_up_plus_alphaCdx)
			.def_readwrite(
					"primal_residual_in_scaled_low_plus_alphaCdx",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_low_plus_alphaCdx)
			.def_readwrite("CTz", &qp::QPWorkspace<f64>::CTz);

	::pybind11::class_<qp::QPResults<f64>>(m, "QPResults")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
																							 // read-write public data member

			.def_readwrite("x", &qp::QPResults<f64>::x)
			.def_readwrite("y", &qp::QPResults<f64>::y)
			.def_readwrite("z", &qp::QPResults<f64>::z)
			.def_readwrite("n_c", &qp::QPResults<f64>::n_c)
			.def_readwrite("mu_eq", &qp::QPResults<f64>::mu_eq)
			.def_readwrite("mu_eq_inv", &qp::QPResults<f64>::mu_eq_inv)
			.def_readwrite("mu_in", &qp::QPResults<f64>::mu_in)
			.def_readwrite("mu_in_inv", &qp::QPResults<f64>::mu_in_inv)
			.def_readwrite("rho", &qp::QPResults<f64>::rho)
			.def_readwrite("n_tot", &qp::QPResults<f64>::n_tot)
			.def_readwrite("n_ext", &qp::QPResults<f64>::n_ext)
			.def_readwrite("timing", &qp::QPResults<f64>::timing)
			.def_readwrite("objValue", &qp::QPResults<f64>::objValue)
			.def_readwrite("n_mu_change", &qp::QPResults<f64>::n_mu_change);

	::pybind11::class_<qp::QPSettings<f64>>(m, "QPSettings")
			.def(::pybind11::init()) // constructor
															 // read-write public data member

			.def_readwrite("alpha_bcl", &qp::QPSettings<f64>::alpha_bcl)
			.def_readwrite("beta_bcl", &qp::QPSettings<f64>::beta_bcl)
			.def_readwrite("refactor_dual_feasibility_threshold", &qp::QPSettings<f64>::refactor_dual_feasibility_threshold)
			.def_readwrite("pmm", &qp::QPSettings<f64>::pmm)
			.def_readwrite(
					"refactor_rho_threshold",
					&qp::QPSettings<f64>::refactor_rho_threshold)
			.def_readwrite("mu_max_eq", &qp::QPSettings<f64>::mu_max_eq)
			.def_readwrite("mu_max_in", &qp::QPSettings<f64>::mu_max_in)
			.def_readwrite("mu_max_eq_inv", &qp::QPSettings<f64>::mu_max_eq_inv)
			.def_readwrite("mu_max_in_inv", &qp::QPSettings<f64>::mu_max_in_inv)
			.def_readwrite("mu_update_factor", &qp::QPSettings<f64>::mu_update_factor)
			.def_readwrite(
					"mu_update_inv_factor", &qp::QPSettings<f64>::mu_update_inv_factor)

			.def_readwrite("cold_reset_mu_eq", &qp::QPSettings<f64>::cold_reset_mu_eq)
			.def_readwrite("cold_reset_mu_in", &qp::QPSettings<f64>::cold_reset_mu_in)
			.def_readwrite(
					"cold_reset_mu_eq_inv", &qp::QPSettings<f64>::cold_reset_mu_eq_inv)

			.def_readwrite(
					"cold_reset_mu_in_inv", &qp::QPSettings<f64>::cold_reset_mu_in_inv)
			.def_readwrite("max_iter", &qp::QPSettings<f64>::max_iter)
			.def_readwrite("max_iter_in", &qp::QPSettings<f64>::max_iter_in)

			.def_readwrite("eps_abs", &qp::QPSettings<f64>::eps_abs)
			.def_readwrite("eps_rel", &qp::QPSettings<f64>::eps_rel)
			.def_readwrite("eps_IG", &qp::QPSettings<f64>::eps_IG)
			.def_readwrite("R", &qp::QPSettings<f64>::R)
			.def_readwrite(
					"nb_iterative_refinement",
					&qp::QPSettings<f64>::nb_iterative_refinement)
			.def_readwrite("verbose", &qp::QPSettings<f64>::verbose);

	::pybind11::class_<qp::QPData<f64>>(m, "QPData")
			//.def(::pybind11::init()) // constructor
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member

			.def_readonly("H", &qp::QPData<f64>::H)
			.def_readonly("g", &qp::QPData<f64>::g)
			.def_readonly("A", &qp::QPData<f64>::A)
			.def_readonly("b", &qp::QPData<f64>::b)
			.def_readonly("C", &qp::QPData<f64>::C)
			.def_readonly("u", &qp::QPData<f64>::u)
			.def_readonly("l", &qp::QPData<f64>::l)
			.def_readonly("dim", &qp::QPData<f64>::dim)
			.def_readonly("n_eq", &qp::QPData<f64>::n_eq)
			.def_readonly("n_in", &qp::QPData<f64>::n_in)
			.def_readonly("n_total", &qp::QPData<f64>::n_total);

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
	m.def("QPsolve", &qp::pybind11::QPsolve<f32, c>);
	m.def("QPsolve", &qp::pybind11::QPsolve<f64, c>);

	m.def("QPsolveEq", &qp::pybind11::QPsolveEq<f32, c>);
	m.def("QPsolveEq", &qp::pybind11::QPsolveEq<f64, c>);

	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f32, c>);
	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f64, c>);

	m.def("QPsetup", &qp::detail::QPsetup<f32>);
	m.def("QPsetup", &qp::detail::QPsetup<f64>);

	m.def("QPreset", &qp::pybind11::QPreset<f32>);
	m.def("QPreset", &qp::pybind11::QPreset<f64>);

	/*
	m.def("correction_guess", &qp::pybind11::correction_guess<f32>);
	m.def("correction_guess", &qp::pybind11::correction_guess<f64>);
	m.def("initial_guess_fact", &qp::pybind11::initial_guess_fact<f32>);
	m.def("initial_guess_fact", &qp::pybind11::initial_guess_fact<f64>);
	m.def("iterative_solve_with_permut_fact",
	&qp::pybind11::iterative_solve_with_permut_fact<f32>);
	m.def("iterative_solve_with_permut_fact",
	&qp::pybind11::iterative_solve_with_permut_fact<f64>);
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
	m.def("gradient_norm_computation",
	&qp::pybind11::gradient_norm_computation<f32>);
	m.def("gradient_norm_computation",
	&qp::pybind11::gradient_norm_computation<f64>); m.def("transition_algebra",
	&qp::pybind11::transition_algebra<f32>); m.def("transition_algebra",
	&qp::pybind11::transition_algebra<f64>);

	m.def("transition_algebra_before_LS_CG",
	&qp::pybind11::transition_algebra_before_LS_CG<f32>);
	m.def("transition_algebra_before_LS_CG",
	&qp::pybind11::transition_algebra_before_LS_CG<f64>);
	m.def("transition_algebra_after_LS_CG",
	&qp::pybind11::transition_algebra_after_LS_CG<f32>);
	m.def("transition_algebra_after_LS_CG",
	&qp::pybind11::transition_algebra_after_LS_CG<f64>);
	m.def("transition_algebra_before_IG_newton",
	&qp::pybind11::transition_algebra_before_IG_newton<f32>);
	m.def("transition_algebra_before_IG_newton",
	&qp::pybind11::transition_algebra_before_IG_newton<f64>);
	m.def("transition_algebra_after_IG_LS",
	&qp::pybind11::transition_algebra_after_IG_LS<f32>);
	m.def("transition_algebra_after_IG_LS",
	&qp::pybind11::transition_algebra_after_IG_LS<f64>);
	*/

	m.attr("__version__") = "dev";
}
