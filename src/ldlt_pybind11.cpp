
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

		T eps_abs = 1.e-9,
		T eps_rel = 0,
		const bool VERBOSE = true

		) {

	qpsettings.eps_abs = eps_abs;
	qpsettings.eps_rel = eps_rel;
	qpsettings.verbose = VERBOSE;

	qpmodel._H = H.eval();
	qpmodel._g = g.eval();
    qpmodel._A = A.eval();
    qpmodel._b = b.eval();
    qpmodel._C = C.eval();
    qpmodel._u = u.eval();
    qpmodel._l = l.eval();

	qpwork.h_scaled = qpmodel._H;
	qpwork.g_scaled = qpmodel._g;
	qpwork.a_scaled = qpmodel._A;
	qpwork.b_scaled = qpmodel._b;
    qpwork.c_scaled = qpmodel._C;
    qpwork.u_scaled = qpmodel._u;
    qpwork.l_scaled = qpmodel._l;

    qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen,qpwork.h_scaled},
			{from_eigen,qpwork.g_scaled},
			{from_eigen,qpwork.a_scaled},
			{from_eigen,qpwork.b_scaled},
			{from_eigen,qpwork.c_scaled},
			{from_eigen,qpwork.u_scaled},
			{from_eigen,qpwork.l_scaled}
	};

	qpwork.ruiz.scale_qp_in_place(qp_scaled,VectorViewMut<T>{from_eigen,qpwork.dw_aug});
    qpwork.dw_aug.setZero();

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
    qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
    qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel._g);
	qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qpwork.h_scaled ;
	qpwork.kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults.rho;
	qpwork.kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qpwork.a_scaled.transpose();
	qpwork.kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qpwork.a_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
	qpwork.kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults.mu_eq_inv); // mu stores the inverse of mu

	qpwork.ldl.factorize(qpwork.kkt);
	qpwork.rhs.head(qpmodel._dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel._dim,qpmodel._n_eq) = qpwork.b_scaled;

    qp::detail::iterative_solve_with_permut_fact( //
		qpsettings,
		qpmodel,
		qpresults,
		qpwork,
		T(1),
		qpmodel._dim+qpmodel._n_eq
        );

	qpresults.x = qpwork.dw_aug.head(qpmodel._dim);
	qpresults.y = qpwork.dw_aug.segment(qpmodel._dim,qpmodel._n_eq);

	qpwork.dw_aug.setZero();

}

template <typename T, Layout L>
void QPupdateMatrice( //
		MatRef<T, L> H,
		MatRef<T, L> A,
		MatRef<T, L> C,
		const qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		const bool update = false
		) {

		// update matrices

		qpmodel._H = H.eval();
		qpmodel._A = A.eval();
		qpmodel._C = C.eval();

		qpwork.h_scaled = qpmodel._H;
		qpwork.g_scaled = qpmodel._g;
		qpwork.a_scaled = qpmodel._A;
		qpwork.b_scaled = qpmodel._b;
		qpwork.c_scaled = qpmodel._C;
		qpwork.u_scaled = qpmodel._u;
		qpwork.l_scaled = qpmodel._l;

		qp::QpViewBoxMut<T> qp_scaled{
				{from_eigen,qpwork.h_scaled},
				{from_eigen,qpwork.g_scaled},
				{from_eigen,qpwork.a_scaled},
				{from_eigen,qpwork.b_scaled},
				{from_eigen,qpwork.c_scaled},
				{from_eigen,qpwork.u_scaled},
				{from_eigen,qpwork.l_scaled}
		};

		qpwork.ruiz.scale_qp_in_place(qp_scaled,VectorViewMut<T>{from_eigen,qpwork.dw_aug});
		qpwork.dw_aug.setZero();

		// re update all other variables
		if (update){
			qpresults.clearResults();
		}

		// perform warm start

		qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
		qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
		qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
		qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel._g);
		qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);

		qpwork.kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qpwork.h_scaled ;
		qpwork.kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults.rho;
		qpwork.kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qpwork.a_scaled.transpose();
		qpwork.kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qpwork.a_scaled;
		qpwork.kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
		qpwork.kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults.mu_eq_inv); // mu stores the inverse of mu

		qpwork.ldl.factorize(qpwork.kkt);
		qpwork.rhs.head(qpmodel._dim) = -qpwork.g_scaled;
		qpwork.rhs.segment(qpmodel._dim,qpmodel._n_eq) = qpwork.b_scaled;

		qp::detail::iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			T(1),
			qpmodel._dim+qpmodel._n_eq
			);

		qpresults.x = qpwork.dw_aug.head(qpmodel._dim);
		qpresults.y = qpwork.dw_aug.segment(qpmodel._dim,qpmodel._n_eq);

		qpwork.dw_aug.setZero();

}

template <typename T>
void QPupdateVectors( //
		VecRef<T> g,
		VecRef<T> b,
		VecRef<T> u,
		VecRef<T> l,
		const qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		const bool update = false
		) {

		// update vectors

		qpmodel._g = g.eval();
		qpmodel._b = b.eval();
		qpmodel._u = u.eval();
		qpmodel._l = l.eval();

		qpwork.g_scaled = qpmodel._g;
		qpwork.b_scaled = qpmodel._b;
		qpwork.u_scaled = qpmodel._u;
		qpwork.l_scaled = qpmodel._l;

		qpwork.ruiz.scale_primal_in_place(VectorViewMut<T>{from_eigen,qpwork.g_scaled});
		qpwork._scale_dual_in_place_eq(VectorViewMut<T>{from_eigen,qpwork.b_scaled});
		qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen,qpwork.u_scaled});
		qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen,qpwork.l_scaled});

		// re update all other variables --> no need for warm start (reuse previous one ?) ? to finish

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

	qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork.dual_residual_scaled});
	qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork.CTz});

	qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
	qpwork._primal_residual_in_scaled_u -= qpmodel._u;
	qpwork._primal_residual_in_scaled_l -= qpmodel._l;

	qpwork.ruiz.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.ze});
	qpwork._primal_residual_in_scaled_u += qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu

	qpwork._l_active_set_n_u.array() = (qpwork._primal_residual_in_scaled_u .array() >= 0);
	qpwork._l_active_set_n_l.array() = (qpwork._primal_residual_in_scaled_l.array() <= 0);

	qpwork.active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpwork._primal_residual_in_scaled_u -= qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l -= qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu

	qpwork.ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u});
	qpwork.ruiz.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_l});
	qpwork.ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.ze});
	// rescale value
	isize numactive_inequalities = qpwork.active_inequalities.count();
	isize inner_pb_dim = qpmodel._dim + qpmodel._n_eq + numactive_inequalities;


	qp::line_search::active_set_change_new(
			qpwork,
			qpresults,
			qpmodel);

	qpwork.err.head(inner_pb_dim).setZero();
	qpwork.rhs.head(qpmodel._dim).setZero();

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork.rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_u(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork.rhs(j + qpmodel._dim + qpmodel._n_eq) = -qpwork._primal_residual_in_scaled_l(i);
			}
		} else {
			qpwork.rhs.head(qpmodel._dim).noalias() += qpresults.z(i) * qpwork.c_scaled.row(i); // add CTze_inactif to rhs.head(dim)
		}
	}

	qpwork.rhs.head(qpmodel._dim) = -qpwork.dual_residual_scaled; // rhs.head(dim) contains now : -(Hxe + g + ATye + CTze_actif)
	qpwork.rhs.segment(qpmodel._dim, qpmodel._n_eq) = -qpwork.primal_residual_eq_scaled;

	{
	detail::iterative_solve_with_permut_fact( //
			qpwork,
			qpresults,
			qpmodel,
			qpsettings,
			eps_int,
			inner_pb_dim);
	}

	qpwork._d_dual_for_eq = qpwork.rhs.head(qpmodel._dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz_actif by definition of the solution

	for (isize i = 0; i < qpmodel._n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults._n_c) {
			if (qpwork._l_active_set_n_u(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork.dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork.c_scaled.row(i);
			} else if (qpwork._l_active_set_n_l(i)) {
				qpwork._d_dual_for_eq.noalias() -= qpwork.dw_aug(j + qpmodel._dim + qpmodel._n_eq) * qpwork.c_scaled.row(i);
			}
		}
	}

	// use active_part_z as a temporary variable to permut back dw_aug newton step
	for (isize j = 0; j < qpmodel._n_in; ++j) {
		isize i = qpwork.current_bijection_map(j);
		if (i < qpresults._n_c) {
			//dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			//cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;

			qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel._dim + qpmodel._n_eq + i);
			qpwork.Cdx(j) = qpwork.rhs(i + qpmodel._dim + qpmodel._n_eq) + qpwork.dw_aug(qpmodel._dim + qpmodel._n_eq + i) * qpresults.mu_in_inv; // mu stores the inverse of mu

		} else {
			//dw_aug_(j + dim + n_eq) = -z_(j);
			qpwork.active_part_z(j) = -qpresults.z(j);
			qpwork.Cdx(j) = qpwork.c_scaled.row(j).dot(qpwork.dw_aug.head(qpmodel._dim));
		}
	}
	qpwork.dw_aug.tail(qpmodel._n_in) = qpwork.active_part_z ;

	qpwork._primal_residual_in_scaled_u += qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu
	qpwork._primal_residual_in_scaled_l += qpwork.ze * qpresults.mu_in_inv; // mu stores the inverse of mu

	//qpwork._d_primal_residual_eq = qpwork.rhs.segment(qpmodel._dim, qpmodel._n_eq); // By definition of linear system solution // seems unprecise
	qpwork._d_primal_residual_eq.noalias() = qpwork.a_scaled * qpwork.dw_aug.head(qpmodel._dim);

	qpwork.dual_residual_scaled -= qpwork.CTz; // contains now Hxe+g+ATye

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

		const bool do_initial_guess_fact = primal_feasibility_lhs < qpsettings.eps_IG || qpmodel._n_in == 0;
		bool do_correction_guess = (!do_initial_guess_fact && qpmodel._n_in != 0) ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in && qpmodel._n_in != 0) ;

		if (do_initial_guess_fact && err_in >= bcl_eta_in ) {

			//
			// ATy contains : Hx_new + rho*(x_new-xe) + ATy_new
			// primal_residual_eq_scaled contains : Ax_new - b -(y_new-ye)//mu_eq
			// Hence ATy becomes below as wanted : Hx_new + rho*(x_new-xe) + mu_eq * AT(Ax_new-b + ye/mu_eq)
			//
			qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});
			qpwork._Hx.noalias() += qpwork.alpha * qpwork.h_scaled * qpwork.dw_aug.head(qpmodel._dim);
			qpwork._ATy.noalias() +=  (qpwork.a_scaled.transpose() * qpwork.primal_residual_eq_scaled) * qpresults.mu_eq ; //mu stores mu
			qpwork.primal_residual_eq_scaled.noalias() += qpresults.y * qpresults.mu_eq_inv ; // contains now Ax_new - b + ye/mu_eq
			qpwork.active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork.dual_residual_scaled.noalias() = qpwork._ATy;
			qpwork.dual_residual_scaled.noalias() +=  qpwork.c_scaled.transpose() * qpwork.active_part_z * qpresults.mu_in ; //mu stores mu  // used for newton step at first iteration

			qpwork._primal_residual_in_scaled_u.noalias() += qpresults.z * qpresults.mu_in_inv; //mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l.noalias() += qpresults.z * qpresults.mu_in_inv; //mu stores the inverse of mu
		}
		if (!do_initial_guess_fact ) {

			qpwork.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{from_eigen, qpwork._primal_residual_in_scaled_u}); // primal_residual_in_scaled_u contains Cx unscaled from global primal residual
			qpwork.primal_residual_eq_scaled.noalias()  += qpwork.ye * qpresults.mu_eq_inv;//mu stores the inverse of mu
			qpwork._ATy.noalias() =  (qpwork.a_scaled.transpose() * qpwork.primal_residual_eq_scaled) * qpresults.mu_eq ; //mu stores mu
			qpwork.ruiz.scale_dual_residual_in_place(VectorViewMut<T>{from_eigen,qpwork._Hx});

			qpwork._primal_residual_in_scaled_u.noalias()  += qpwork.ze * qpresults.mu_in_inv;//mu stores the inverse of mu
			qpwork._primal_residual_in_scaled_l = qpwork._primal_residual_in_scaled_u;
			qpwork._primal_residual_in_scaled_u -= qpwork.u_scaled;
			qpwork._primal_residual_in_scaled_l -= qpwork.l_scaled;
			qpwork.dual_residual_scaled.noalias() = qpwork._Hx + qpwork._ATy + qpwork.g_scaled;
			qpwork.dual_residual_scaled.noalias() += qpresults._rho * (qpresults.x - qpwork.xe);
			qpwork.active_part_z = qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l);
			qpwork.dual_residual_scaled.noalias() +=  qpwork.c_scaled.transpose() * qpwork.active_part_z * qpresults.mu_in ; //mu stores mu  // used for newton step at first iteration

		}

}


template <typename T>
void transition_algebra_before_LS_CG(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpdata<T>& qpmodel){

		qpwork._d_dual_for_eq.noalias() = qpwork.h_scaled * qpwork.dw_aug.head(qpmodel._dim);
		//qpwork._d_primal_residual_eq.noalias() = qpwork.dw_aug.segment(qpmodel._dim,qpmodel._n_eq) * qpresults.mu_eq_inv; // by definition Adx = dy / mu : seems unprecise
		qpwork._d_primal_residual_eq.noalias() = qpwork.a_scaled * qpwork.dw_aug.head(qpmodel._dim);
		qpwork.Cdx.noalias() = qpwork.c_scaled * qpwork.dw_aug.head(qpmodel._dim);

}

template <typename T>
T transition_algebra_after_LS_CG(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel){

		qpresults.x.noalias() += qpwork.alpha * qpwork.dw_aug.head(qpmodel._dim);

		qpwork._primal_residual_in_scaled_u.noalias() += qpwork.alpha * qpwork.Cdx;
		qpwork._primal_residual_in_scaled_l.noalias() += qpwork.alpha * qpwork.Cdx;
		qpwork.primal_residual_eq_scaled.noalias() += qpwork.alpha * qpwork._d_primal_residual_eq;
		qpresults.y.noalias() = qpwork.primal_residual_eq_scaled * qpresults.mu_eq; //mu stores mu

		qpwork._Hx.noalias() += qpwork.alpha * qpwork._d_dual_for_eq ; // stores Hx
		qpwork._ATy.noalias() = (qpwork.a_scaled).transpose() * qpresults.y ;

		qpresults.z =  (qp::detail::positive_part(qpwork._primal_residual_in_scaled_u) + qp::detail::negative_part(qpwork._primal_residual_in_scaled_l)) *  qpresults.mu_in; //mu stores mu
		T rhs_c = std::max(qpwork.correction_guess_rhs_g, infty_norm( qpwork._Hx));
		rhs_c = std::max(rhs_c, infty_norm(qpwork._ATy));
		qpwork.dual_residual_scaled.noalias() = qpwork.c_scaled.transpose() * qpresults.z ;
		rhs_c = std::max(rhs_c, infty_norm(qpwork.dual_residual_scaled));
		qpwork.dual_residual_scaled.noalias() += qpwork._Hx + qpwork.g_scaled + qpwork._ATy + qpresults._rho * (qpresults.x - qpwork.xe);
		std::cout << "rhs_c " << rhs_c  << std::endl;
		T err_in = infty_norm(qpwork.dual_residual_scaled);
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

	qpwork._primal_residual_in_scaled_u += (qpwork.alpha * qpwork.Cdx);
	qpwork._primal_residual_in_scaled_l += (qpwork.alpha * qpwork.Cdx);
	qpwork._l_active_set_n_u = (qpwork._primal_residual_in_scaled_u.array() >= 0).matrix();
	qpwork._l_active_set_n_l = (qpwork._primal_residual_in_scaled_l.array() <= 0).matrix();
	qpwork.active_inequalities = qpwork._l_active_set_n_u || qpwork._l_active_set_n_l;

	qpresults.x.noalias() += qpwork.alpha * qpwork.dw_aug.head(qpmodel._dim);
	qpresults.y.noalias() += qpwork.alpha * qpwork.dw_aug.segment(qpmodel._dim, qpmodel._n_eq);


	qpwork.active_part_z = qpresults.z + qpwork.alpha * qpwork.dw_aug.tail(qpmodel._n_in) ;

	qpwork._residual_in_z_u_plus_alpha = (qpwork.active_part_z.array() > 0).select(qpwork.active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	qpwork._residual_in_z_l_plus_alpha = (qpwork.active_part_z.array() < 0).select(qpwork.active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));

	qpresults.z = (qpwork._l_active_set_n_u).select(qpwork._residual_in_z_u_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (qpwork._l_active_set_n_l).select(qpwork._residual_in_z_l_plus_alpha, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) +
				   (!qpwork._l_active_set_n_l.array() && !qpwork._l_active_set_n_u.array()).select(qpwork.active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) ;

	qpwork.primal_residual_eq_scaled.noalias() += qpwork.alpha * qpwork._d_primal_residual_eq;
	qpwork.dual_residual_scaled.noalias() += qpwork.alpha * qpwork._d_dual_for_eq;

	qpwork._ATy = qpwork.dual_residual_scaled ;  // will be used in correction guess if needed : contains Hx_new + rho*(x_new-xe) + g + ATynew

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
void QPsolve(
		const qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork,
		const qp::Qpsettings<T>& qpsettings){

			auto start = std::chrono::high_resolution_clock::now();
			qp::detail::qp_solve( //
								qpsettings,
								qpmodel,
								qpresults,
								qpwork);
			auto stop = std::chrono::high_resolution_clock::now();
    		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

			std::cout << "------ SOLVER STATISTICS--------" << std::endl;
			std::cout << "n_ext : " <<  qpresults.n_ext << std::endl;
			std::cout << "n_tot : " <<  qpresults.n_tot << std::endl;
			std::cout << "mu updates : " <<  qpresults.n_mu_change << std::endl;
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
	::pybind11::class_<qp::Qpworkspace<f64>>(m, "Qpworkspace")
        .def(::pybind11::init<i64, i64, i64 &>()) // constructor
        // read-write public data member
        //.def_readwrite("ruiz", &qp::Qpworkspace<f64>::ruiz)
		//.def_readonly("ldl", &qp::Qpworkspace<f64>::ldl)
		.def_readwrite("h_scaled", &qp::Qpworkspace<f64>::h_scaled)
		.def_readwrite("g_scaled", &qp::Qpworkspace<f64>::g_scaled)
		.def_readwrite("a_scaled", &qp::Qpworkspace<f64>::a_scaled)
		.def_readwrite("c_scaled", &qp::Qpworkspace<f64>::c_scaled)
		.def_readwrite("b_scaled", &qp::Qpworkspace<f64>::b_scaled)
		.def_readwrite("u_scaled", &qp::Qpworkspace<f64>::u_scaled)
		.def_readwrite("l_scaled", &qp::Qpworkspace<f64>::l_scaled)
		.def_readwrite("x_prev", &qp::Qpworkspace<f64>::x_prev)
		.def_readwrite("y_prev", &qp::Qpworkspace<f64>::y_prev)
		.def_readwrite("z_prev", &qp::Qpworkspace<f64>::z_prev)
		.def_readwrite("kkt", &qp::Qpworkspace<f64>::kkt)
		.def_readwrite("current_bijection_map", &qp::Qpworkspace<f64>::current_bijection_map)
		.def_readwrite("new_bijection_map", &qp::Qpworkspace<f64>::new_bijection_map)
		.def_readwrite("active_set_up", &qp::Qpworkspace<f64>::active_set_up)
		.def_readwrite("active_set_low", &qp::Qpworkspace<f64>::active_set_low)
		.def_readwrite("active_inequalities", &qp::Qpworkspace<f64>::active_inequalities)
		.def_readwrite("Hdx", &qp::Qpworkspace<f64>::Hdx)
		.def_readwrite("Cdx", &qp::Qpworkspace<f64>::Cdx)
		.def_readwrite("Adx", &qp::Qpworkspace<f64>::Adx)
		.def_readwrite("active_part_z", &qp::Qpworkspace<f64>::active_part_z)
		.def_readwrite("alphas", &qp::Qpworkspace<f64>::alphas)
		.def_readwrite("dw_aug", &qp::Qpworkspace<f64>::dw_aug)
		.def_readwrite("rhs", &qp::Qpworkspace<f64>::rhs)
		.def_readwrite("err", &qp::Qpworkspace<f64>::err)
		.def_readwrite("primal_feasibility_rhs_1_eq", &qp::Qpworkspace<f64>::primal_feasibility_rhs_1_eq)
		.def_readwrite("primal_feasibility_rhs_1_in_u", &qp::Qpworkspace<f64>::primal_feasibility_rhs_1_in_u)
		.def_readwrite("primal_feasibility_rhs_1_in_l", &qp::Qpworkspace<f64>::primal_feasibility_rhs_1_in_l)
		.def_readwrite("dual_feasibility_rhs_2", &qp::Qpworkspace<f64>::dual_feasibility_rhs_2)
		.def_readwrite("correction_guess_rhs_g", &qp::Qpworkspace<f64>::correction_guess_rhs_g)
		.def_readwrite("alpha", &qp::Qpworkspace<f64>::alpha)
		.def_readwrite("dual_residual_scaled", &qp::Qpworkspace<f64>::dual_residual_scaled)
		.def_readwrite("primal_residual_eq_scaled", &qp::Qpworkspace<f64>::primal_residual_eq_scaled)
		.def_readwrite("primal_residual_in_scaled_up", &qp::Qpworkspace<f64>::primal_residual_in_scaled_up)
		.def_readwrite("primal_residual_in_scaled_low", &qp::Qpworkspace<f64>::primal_residual_in_scaled_low)
		.def_readwrite("primal_residual_in_scaled_up_plus_alphaCdx", &qp::Qpworkspace<f64>::primal_residual_in_scaled_up_plus_alphaCdx)
		.def_readwrite("primal_residual_in_scaled_low_plus_alphaCdx", &qp::Qpworkspace<f64>::primal_residual_in_scaled_low_plus_alphaCdx)
		.def_readwrite("CTz", &qp::Qpworkspace<f64>::CTz);

	::pybind11::class_<qp::Qpresults<f64>>(m, "Qpresults")
        .def(::pybind11::init<i64, i64, i64 &>()) // constructor
        // read-write public data member

		.def_readwrite("x", &qp::Qpresults<f64>::x)
		.def_readwrite("y", &qp::Qpresults<f64>::y)
		.def_readwrite("z", &qp::Qpresults<f64>::z)
		.def_readwrite("n_c", &qp::Qpresults<f64>::n_c)
		.def_readwrite("mu_eq", &qp::Qpresults<f64>::mu_eq)
		.def_readwrite("mu_eq_inv", &qp::Qpresults<f64>::mu_eq_inv)
		.def_readwrite("mu_in", &qp::Qpresults<f64>::mu_in)
		.def_readwrite("mu_in_inv", &qp::Qpresults<f64>::mu_in_inv)
		.def_readwrite("rho", &qp::Qpresults<f64>::rho)
		.def_readwrite("n_tot", &qp::Qpresults<f64>::n_tot)
		.def_readwrite("n_ext", &qp::Qpresults<f64>::n_ext)
		.def_readwrite("timing", &qp::Qpresults<f64>::timing)
		.def_readwrite("objValue", &qp::Qpresults<f64>::objValue)
		.def_readwrite("n_mu_change", &qp::Qpresults<f64>::n_mu_change);


	::pybind11::class_<qp::Qpsettings<f64>>(m, "Qpsettings")
        .def(::pybind11::init()) // constructor
        // read-write public data member

		.def_readwrite("alpha_bcl", &qp::Qpsettings<f64>::alpha_bcl)
		.def_readwrite("beta_bcl", &qp::Qpsettings<f64>::beta_bcl)
		.def_readwrite("refactor_dual_feasibility_threshold", &qp::Qpsettings<f64>::refactor_dual_feasibility_threshold)
		.def_readwrite("refactor_rho_threshold", &qp::Qpsettings<f64>::refactor_rho_threshold)
		.def_readwrite("mu_max_eq", &qp::Qpsettings<f64>::mu_max_eq)
		.def_readwrite("mu_max_in", &qp::Qpsettings<f64>::mu_max_in)
		.def_readwrite("mu_max_eq_inv", &qp::Qpsettings<f64>::mu_max_eq_inv)
		.def_readwrite("mu_max_in_inv", &qp::Qpsettings<f64>::mu_max_in_inv)
		.def_readwrite("mu_update_factor", &qp::Qpsettings<f64>::mu_update_factor)
		.def_readwrite("mu_update_inv_factor", &qp::Qpsettings<f64>::mu_update_inv_factor)

		.def_readwrite("cold_reset_mu_eq", &qp::Qpsettings<f64>::cold_reset_mu_eq)
		.def_readwrite("cold_reset_mu_in", &qp::Qpsettings<f64>::cold_reset_mu_in)
		.def_readwrite("cold_reset_mu_eq_inv", &qp::Qpsettings<f64>::cold_reset_mu_eq_inv)

		.def_readwrite("cold_reset_mu_in_inv", &qp::Qpsettings<f64>::cold_reset_mu_in_inv)
		.def_readwrite("max_iter", &qp::Qpsettings<f64>::max_iter)
		.def_readwrite("max_iter_in", &qp::Qpsettings<f64>::max_iter_in)

		.def_readwrite("eps_abs", &qp::Qpsettings<f64>::eps_abs)
		.def_readwrite("eps_rel", &qp::Qpsettings<f64>::eps_rel)
		.def_readwrite("_err_IG", &qp::Qpsettings<f64>::eps_IG)
		.def_readwrite("R", &qp::Qpsettings<f64>::R)
		.def_readwrite("nb_iterative_refinement", &qp::Qpsettings<f64>::nb_iterative_refinement)
		.def_readwrite("verbose", &qp::Qpsettings<f64>::verbose);

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
	m.def("QPsolve", &qp::pybind11::QPsolve<f32, c>);
	m.def("QPsolve", &qp::pybind11::QPsolve<f64, c>);

	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f32, c>);
	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f64, c>);

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
