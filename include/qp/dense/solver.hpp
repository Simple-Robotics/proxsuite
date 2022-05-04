#ifndef PROXSUITE_INCLUDE_QP_DENSE_SOLVER_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_SOLVER_HPP

#include "qp/dense/views.hpp"
#include "qp/dense/linesearch.hpp"
#include "qp/dense/utils.hpp"
#include <cmath>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <veg/util/dynstack_alloc.hpp>
#include <dense-ldlt/ldlt.hpp>
#include <fmt/chrono.h>

namespace qp {
namespace dense {

template <typename T>
void refactorize(
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T rho_new) {

	if (!qpwork.constraints_changed && rho_new == qpresults.info.rho) {
		return;
	}

	qpwork.dw_aug.setZero();
	qpwork.kkt.diagonal().head(qpmodel.dim).array() += rho_new - qpresults.info.rho;
	qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
			-qpresults.info.mu_eq;

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
	qpwork.ldl.factorize(qpwork.kkt, stack);

	isize n = qpmodel.dim;
	isize n_eq = qpmodel.n_eq;
	isize n_in = qpmodel.n_in;
	isize n_c = qpresults.info.n_c;

	LDLT_TEMP_MAT(T, new_cols, n + n_eq + n_c, n_c, stack);
	T mu_in_neg = -qpresults.info.mu_in;
	for (isize i = 0; i < n_in; ++i) {
		isize j = qpwork.current_bijection_map[i];
		if (j < n_c) {
			auto col = new_cols.col(j);
			col.head(n) = qpwork.C_scaled.row(i);
			col.segment(n, n_eq + n_c).setZero();
			col(n + n_eq + j) = mu_in_neg;
		}
	}
	qpwork.ldl.insert_block_at(n + n_eq, new_cols, stack);

	qpwork.constraints_changed = false;

	qpwork.dw_aug.setZero();
}

template <typename T>
void mu_update(
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T mu_eq_new,
		T mu_in_new) {
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};

	isize n = qpmodel.dim;
	isize n_eq = qpmodel.n_eq;
	isize n_c = qpresults.info.n_c;

	if ((n_eq + n_c) == 0) {
		return;
	}

	LDLT_TEMP_VEC_UNINIT(T, rank_update_alpha, n_eq + n_c, stack);
	rank_update_alpha.head(n_eq).setConstant(qpresults.info.mu_eq - mu_eq_new);
	rank_update_alpha.tail(n_c).setConstant(qpresults.info.mu_in - mu_in_new);

	{
		auto _indices =
				stack.make_new_for_overwrite(veg::Tag<isize>{}, n_eq + n_c).unwrap();
		isize* indices = _indices.ptr_mut();
		for (isize k = 0; k < n_eq; ++k) {
			indices[k] = n + k;
		}
		for (isize k = 0; k < n_c; ++k) {
			indices[n_eq + k] = n + n_eq + k;
		}
		qpwork.ldl.diagonal_update_clobber_indices(
				indices, n_eq + n_c, rank_update_alpha, stack);
	}

	qpwork.constraints_changed = true;
}

template <typename T>
void iterative_residual(
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		isize inner_pb_dim) {

	qpwork.err.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);

	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.H_scaled.template selfadjointView<Eigen::Lower>() *
			qpwork.dw_aug.head(qpmodel.dim);
	qpwork.err.head(qpmodel.dim) -=
			qpresults.info.rho * qpwork.dw_aug.head(qpmodel.dim);

	// PERF: fuse {A, C}_scaled multiplication operations
	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.A_scaled.transpose() *
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.info.n_c) {
			qpwork.err.head(qpmodel.dim).noalias() -=
					qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
					qpwork.C_scaled.row(i);
			qpwork.err(qpmodel.dim + qpmodel.n_eq + j) -=
					(qpwork.C_scaled.row(i).dot(qpwork.dw_aug.head(qpmodel.dim)) -
			     qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
			         qpresults.info.mu_in);
		}
	}
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() -=
			qpwork.A_scaled *
			qpwork.dw_aug.head(qpmodel.dim);
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) +=
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) *
			qpresults.info.mu_eq;
}

template <typename T>
void iterative_solve_with_permut_fact( //
		const qp::Settings<T>& qpsettings,
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T eps,
		isize inner_pb_dim) {

	qpwork.err.setZero();
	i32 it = 0;
	i32 it_stability = 0;

	qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
	qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), stack);

	iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

	++it;
	T preverr = infty_norm(qpwork.err.head(inner_pb_dim));
	if (qpsettings.verbose) {
		std::cout << "infty_norm(res) " << infty_norm(qpwork.err.head(inner_pb_dim))
							<< std::endl;
	}
	while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

		if (it >= qpsettings.nb_iterative_refinement) {
			break;
		}

		++it;
		qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
		qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

		qpwork.err.head(inner_pb_dim).setZero();
		iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
			it_stability += 1;

		} else {
			it_stability = 0;
		}
		if (it_stability == 2) {
			break;
		}
		preverr = infty_norm(qpwork.err.head(inner_pb_dim));

		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
	}

	if (infty_norm(qpwork.err.head(inner_pb_dim)) >=
	    std::max(eps, qpsettings.eps_refact)) {
		refactorize(qpmodel, qpresults, qpwork, qpresults.info.rho);
		it = 0;
		it_stability = 0;

		qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
		qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), stack);

		iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		preverr = infty_norm(qpwork.err.head(inner_pb_dim));
		++it;
		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
		while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

			if (it >= qpsettings.nb_iterative_refinement) {
				break;
			}
			++it;
			qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), stack);
			qpwork.dw_aug.head(inner_pb_dim) += qpwork.err.head(inner_pb_dim);

			qpwork.err.head(inner_pb_dim).setZero();
			iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

			if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
				it_stability += 1;

			} else {
				it_stability = 0;
			}
			if (it_stability == 2) {
				break;
			}
			preverr = infty_norm(qpwork.err.head(inner_pb_dim));

			if (qpsettings.verbose) {
				std::cout << "infty_norm(res) "
									<< infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
			}
		}
	}
	qpwork.rhs.head(inner_pb_dim).setZero();
}

template <typename T>
void bcl_update(
		const qp::Settings<T>& qpsettings,
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T& primal_feasibility_lhs_new,
		T& bcl_eta_ext,
		T& bcl_eta_in,

		T bcl_eta_ext_init,
		T eps_in_min,

		T& new_bcl_mu_in,
		T& new_bcl_mu_eq,
		T& new_bcl_mu_in_inv,
		T& new_bcl_mu_eq_inv

) {
	if (primal_feasibility_lhs_new <= bcl_eta_ext) {
		if (qpsettings.verbose) {
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext * pow(qpresults.info.mu_in, qpsettings.beta_bcl);
		bcl_eta_in = std::max(bcl_eta_in * qpresults.info.mu_in, eps_in_min);
	} else {
		if (qpsettings.verbose) {
			std::cout << "bad step" << std::endl;
		}

		qpresults.y = qpwork.y_prev;
		qpresults.z = qpwork.z_prev;

		new_bcl_mu_in = std::max(
				qpresults.info.mu_in * qpsettings.mu_update_factor, qpsettings.mu_max_in);
		new_bcl_mu_eq = std::max(
				qpresults.info.mu_eq * qpsettings.mu_update_factor, qpsettings.mu_max_eq);
		new_bcl_mu_in_inv = std::min(
				qpresults.info.mu_in_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_in_inv); // mu stores the inverse of mu
		new_bcl_mu_eq_inv = std::min(
				qpresults.info.mu_eq_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_eq_inv); // mu stores the inverse of mu
		bcl_eta_ext =
				bcl_eta_ext_init * pow(new_bcl_mu_in, qpsettings.alpha_bcl);
		bcl_eta_in = std::max(new_bcl_mu_in, eps_in_min);
	}
}

template <typename T>
auto compute_inner_loop_saddle_point(
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork) -> T {

	qpwork.active_part_z =
			qp::dense::positive_part(qpwork.primal_residual_in_scaled_up) +
			qp::dense::negative_part(qpwork.primal_residual_in_scaled_low) -
			qpresults.z * qpresults.info.mu_in; // contains now : [Cx-u+z_prev*mu_in]+
	                                       // + [Cx-l+z_prev*mu_in]- - z*mu_in

	T err = infty_norm(qpwork.active_part_z);
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) =
			qpwork.primal_residual_eq_scaled; // contains now Ax-b-(y-y_prev)/mu

	T prim_eq_e = infty_norm(
			qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)); // ||Ax-b-(y-y_prev)/mu||
	err = std::max(err, prim_eq_e);
	T dual_e =
			infty_norm(qpwork.dual_residual_scaled); // contains ||Hx + rho(x-xprev) +
	                                             // g + Aty + Ctz||
	err = std::max(err, dual_e);

	return err;
}

template <typename T>
void primal_dual_semi_smooth_newton_step(
		const qp::Settings<T>& qpsettings,
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T eps) {

	/* MUST BE
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+mu_eq (y_prev-y)
	 *  primal_residual_in_scaled_up = Cx-u+mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+mu_in(z_prev)
	 */

	qpwork.active_set_up.array() =
			(qpwork.primal_residual_in_scaled_up.array() >= 0);
	qpwork.active_set_low.array() =
			(qpwork.primal_residual_in_scaled_low.array() <= 0);
	qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
	isize numactive_inequalities = qpwork.active_inequalities.count();

	isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
	qpwork.rhs.setZero();
	qpwork.dw_aug.setZero();

	qp::dense::linesearch::active_set_change(qpmodel, qpresults, qpwork);

	qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled;

	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
			-qpwork.primal_residual_eq_scaled;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.info.n_c) {
			if (qpwork.active_set_up(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_up(i) +
						qpresults.z(i) * qpresults.info.mu_in;
			} else if (qpwork.active_set_low(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_low(i) +
						qpresults.z(i) * qpresults.info.mu_in;
			}
		} else {
			qpwork.rhs.head(qpmodel.dim) +=
					qpresults.z(i) *
					qpwork.C_scaled.row(i); // unactive unrelevant columns
		}
	}

	iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			eps,
			inner_pb_dim);

	// use active_part_z as a temporary variable to derive unpermutted dz step
	for (isize j = 0; j < qpmodel.n_in; ++j) {
		isize i = qpwork.current_bijection_map(j);
		if (i < qpresults.info.n_c) {
			qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
		} else {
			qpwork.active_part_z(j) = -qpresults.z(j);
		}
	}
	qpwork.dw_aug.tail(qpmodel.n_in) = qpwork.active_part_z;
}

template <typename T>
T primal_dual_newton_semi_smooth(
		const qp::Settings<T>& qpsettings,
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		T eps_int) {

	/* MUST CONTAIN IN ENTRY WITH x = x_prev ; y = y_prev ; z = z_prev
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+mu_eq (y_prev-y)
	 *  primal_residual_in_scaled_up = Cx-u+mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+mu_in(z_prev)
	 */

	T err_in = 1.e6;

	for (i64 iter = 0; iter <= qpsettings.max_iter_in; ++iter) {

		if (iter == qpsettings.max_iter_in) {
			qpresults.info.iter += qpsettings.max_iter_in+1;
			break;
		}
		primal_dual_semi_smooth_newton_step<T>(
				qpsettings, qpmodel, qpresults, qpwork, eps_int);

		veg::dynstack::DynStackMut stack{
				veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
		LDLT_TEMP_VEC(T, ATdy, qpmodel.dim, stack);
		LDLT_TEMP_VEC(T, CTdz, qpmodel.dim, stack);

		auto& Hdx = qpwork.Hdx;
		auto& Adx = qpwork.Adx;
		auto& Cdx = qpwork.Cdx;

		auto dx = qpwork.dw_aug.head(qpmodel.dim);
		auto dy = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
		auto dz = qpwork.dw_aug.segment(qpmodel.dim + qpmodel.n_eq, qpmodel.n_in);

		Hdx.setZero();
		Adx.setZero();
		Cdx.setZero();

		Hdx.noalias() +=
				qpwork.H_scaled.template selfadjointView<Eigen::Lower>() * dx;

		Adx.noalias() += qpwork.A_scaled * dx;
		ATdy.noalias() += qpwork.A_scaled.transpose() * dy;

		Cdx.noalias() += qpwork.C_scaled * dx;
		CTdz.noalias() += qpwork.C_scaled.transpose() * dz;

		if (qpmodel.n_in > 0) {
			qp::dense::linesearch::primal_dual_ls(
					qpmodel, qpresults, qpwork, qpsettings);
		}
		auto alpha = qpwork.alpha;

		if (infty_norm(alpha * qpwork.dw_aug) < 1.E-11 && iter > 0) {
			qpresults.info.iter += iter + 1;
			if (qpsettings.verbose) {
				std::cout << "infty_norm(alpha_step * dx) "
									<< infty_norm(alpha * qpwork.dw_aug) << std::endl;
			}
			break;
		}

		qpresults.x += alpha * dx;

		// contains now :  C(x+alpha dx)-u + z_prev * mu_in
		qpwork.primal_residual_in_scaled_up += alpha * Cdx;

		// contains now :  C(x+alpha dx)-l + z_prev * mu_in
		qpwork.primal_residual_in_scaled_low += alpha * Cdx;

		qpwork.primal_residual_eq_scaled +=
				alpha * (Adx - qpresults.info.mu_eq * dy);

		qpresults.y += alpha * dy;
		qpresults.z += alpha * dz;

		qpwork.dual_residual_scaled +=
				alpha * (qpresults.info.rho * dx + Hdx + ATdy + CTdz);

		err_in = dense::compute_inner_loop_saddle_point(qpmodel, qpresults, qpwork);

		if (qpsettings.verbose) {
			std::cout << "---it in " << iter << " projection norm " << err_in
								<< " alpha " << alpha << std::endl;
		}

		if (err_in <= eps_int) {
			qpresults.info.iter += iter + 1;
			break;
		}

		// compute primal and dual infeasibility criteria
		bool is_primal_infeasible = qp::dense::global_primal_residual_infeasibility(
					VectorViewMut<T>{from_eigen,ATdy},
					VectorViewMut<T>{from_eigen,CTdz}, 
					VectorViewMut<T>{from_eigen,dx},
					VectorViewMut<T>{from_eigen,dy},
					VectorViewMut<T>{from_eigen,dz},
					qpwork,
					qpsettings
		);

		bool is_dual_infeasible = qp::dense::global_dual_residual_infeasibility(
					VectorViewMut<T>{from_eigen,Adx},
					VectorViewMut<T>{from_eigen,Cdx}, 
					VectorViewMut<T>{from_eigen,Hdx},
					VectorViewMut<T>{from_eigen,dx},
					qpwork,
					qpsettings,
					qpmodel
		);

		if (is_primal_infeasible){
			qpresults.info.status = PROXQP_PRIMAL_INFEASIBLE;
			break;
		}else if (is_dual_infeasible){
			qpresults.info.status = PROXQP_DUAL_INFEASIBLE;
			break;
		}
	}

	return err_in;
}

template <typename T>
void qp_solve( //
		const qp::Settings<T>& qpsettings,
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork) {

	/*** TEST WITH MATRIX FULL OF NAN FOR DEBUG
	  static constexpr Layout layout = rowmajor;
	  static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	RowMat test(2,2); // test it is full of nan for debug
	std::cout << "test " << test << std::endl;
	*/

	//::Eigen::internal::set_is_malloc_allowed(false);

	T bcl_eta_ext_init = pow(T(0.1), qpsettings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(qpsettings.eps_abs, T(1.E-9));

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);
	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);

	for (i64 iter = 0; iter <= qpsettings.max_iter; ++iter) {

		qpresults.info.iter_ext += 1;
		if (iter == qpsettings.max_iter) {
			break;
		}

		// compute primal residual

		// PERF: fuse matrix product computations in global_{primal, dual}_residual
		qp::dense::global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		qp::dense::global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);
		qpresults.info.pri_res = primal_feasibility_lhs;
		qpresults.info.dua_res = dual_feasibility_lhs;

		T new_bcl_mu_in(qpresults.info.mu_in);
		T new_bcl_mu_eq(qpresults.info.mu_eq);
		T new_bcl_mu_in_inv(qpresults.info.mu_in_inv);
		T new_bcl_mu_eq_inv(qpresults.info.mu_eq_inv);

		T rhs_pri(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_pri +=
					qpsettings.eps_rel *
					std::max(
							std::max(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
							std::max(
									std::max(
											qpwork.primal_feasibility_rhs_1_eq,
											qpwork.primal_feasibility_rhs_1_in_u),
									qpwork.primal_feasibility_rhs_1_in_l));
		}
		bool is_primal_feasible = primal_feasibility_lhs <= rhs_pri;

		T rhs_dua(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_dua +=
					qpsettings.eps_rel *
					std::max(
							std::max(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
							std::max(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2));
		}

		bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;

		if (qpsettings.verbose) {
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext
								<< " bcl_eta_in : " << bcl_eta_in << " rho : " << qpresults.info.rho
								<< " bcl_mu_eq : " << qpresults.info.mu_eq
								<< " bcl_mu_in : " << qpresults.info.mu_in << std::endl;
			std::cout << "qpsettings.eps_abs " << qpsettings.eps_abs
								<< "  qpsettings.eps_rel *rhs "
								<< qpsettings.eps_rel *
											 std::max(
													 std::max(
															 primal_feasibility_eq_rhs_0,
															 primal_feasibility_in_rhs_0),
													 std::max(
															 std::max(
																	 qpwork.primal_feasibility_rhs_1_eq,
																	 qpwork.primal_feasibility_rhs_1_in_u),
															 qpwork.primal_feasibility_rhs_1_in_l))
								<< std::endl;
			std::cout << "is_primal_feasible " << is_primal_feasible
								<< " is_dual_feasible " << is_dual_feasible << std::endl;
		}
		if (is_primal_feasible) {

			if (dual_feasibility_lhs >=
			        qpsettings.refactor_dual_feasibility_threshold &&
			    qpresults.info.rho != qpsettings.refactor_rho_threshold) {

				T rho_new(qpsettings.refactor_rho_threshold);

				refactorize(qpmodel, qpresults, qpwork, rho_new);
				qpresults.info.rho_updates+=1;

				qpresults.info.rho = rho_new;
			}
			if (is_dual_feasible) {
				qpresults.info.status = PROXQP_SOLVED;
				break;
			}
		}

		qpwork.x_prev = qpresults.x;
		qpwork.y_prev = qpresults.y;
		qpwork.z_prev = qpresults.z;

		// primal dual version from gill and robinson

		qpwork.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{
				from_eigen,
				qpwork.primal_residual_in_scaled_up}); // contains now scaled(Cx)
		qpwork.primal_residual_in_scaled_up +=
				qpwork.z_prev *
				qpresults.info.mu_in; // contains now scaled(Cx+z_prev*mu_in)
		qpwork.primal_residual_in_scaled_low = qpwork.primal_residual_in_scaled_up;
		qpwork.primal_residual_in_scaled_up -=
				qpwork.u_scaled; // contains now scaled(Cx-u+z_prev*mu_in)
		qpwork.primal_residual_in_scaled_low -=
				qpwork.l_scaled; // contains now scaled(Cx-l+z_prev*mu_in)

		T err_in = primal_dual_newton_semi_smooth(
				qpsettings, qpmodel, qpresults, qpwork, bcl_eta_in);
		if (qpsettings.verbose) {
			std::cout << " inner loop residual : " << err_in << std::endl;
		}
		if (qpresults.info.status == PROXQP_PRIMAL_INFEASIBLE || qpresults.info.status == PROXQP_DUAL_INFEASIBLE){
			break;
		}

		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::dense::global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		is_primal_feasible =
				primal_feasibility_lhs_new <=
				(qpsettings.eps_abs +
		     qpsettings.eps_rel *
		         std::max(
								 std::max(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 std::max(
										 std::max(
												 qpwork.primal_feasibility_rhs_1_eq,
												 qpwork.primal_feasibility_rhs_1_in_u),
										 qpwork.primal_feasibility_rhs_1_in_l)));
		qpresults.info.pri_res = primal_feasibility_lhs_new;
		if (is_primal_feasible) {
			T dual_feasibility_lhs_new(dual_feasibility_lhs);
			
			qp::dense::global_dual_residual(
					qpmodel,
					qpresults,
					qpwork,
					dual_feasibility_lhs_new,
					dual_feasibility_rhs_0,
					dual_feasibility_rhs_1,
					dual_feasibility_rhs_3);
			qpresults.info.dua_res = dual_feasibility_lhs_new;
			
			is_dual_feasible =
					dual_feasibility_lhs_new <=
					(qpsettings.eps_abs +
			     qpsettings.eps_rel *
			         std::max(
									 std::max(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
									 std::max(
											 dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)));

			if (is_dual_feasible) {
				qpresults.info.status = PROXQP_SOLVED;
				break;
			}
		}

		bcl_update(
				qpsettings,
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				bcl_eta_ext_init,
				eps_in_min,

				new_bcl_mu_in,
				new_bcl_mu_eq,
				new_bcl_mu_in_inv,
				new_bcl_mu_eq_inv

		);

		// COLD RESTART

		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		qp::dense::global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);
		qpresults.info.dua_res = dual_feasibility_lhs_new;

		if (primal_feasibility_lhs_new >= primal_feasibility_lhs &&
		    dual_feasibility_lhs_new >= dual_feasibility_lhs &&
		    qpresults.info.mu_in <= T(1e-5)) {

			if (qpsettings.verbose) {
				std::cout << "cold restart" << std::endl;
			}

			new_bcl_mu_in = qpsettings.cold_reset_mu_in;
			new_bcl_mu_eq = qpsettings.cold_reset_mu_eq;
			new_bcl_mu_in_inv = qpsettings.cold_reset_mu_in_inv;
			new_bcl_mu_eq_inv = qpsettings.cold_reset_mu_eq_inv;
		}

		/// effective mu upddate

		if (qpresults.info.mu_in != new_bcl_mu_in || qpresults.info.mu_eq != new_bcl_mu_eq) {
			{ ++qpresults.info.mu_updates; }
			mu_update(
					qpmodel, qpresults, qpwork, new_bcl_mu_eq, new_bcl_mu_in);
		}

		qpresults.info.mu_eq = new_bcl_mu_eq;
		qpresults.info.mu_in = new_bcl_mu_in;
		qpresults.info.mu_eq_inv = new_bcl_mu_eq_inv;
		qpresults.info.mu_in_inv = new_bcl_mu_in_inv;
	}

	qpwork.ruiz.unscale_primal_in_place(
			VectorViewMut<T>{from_eigen, qpresults.x});
	qpwork.ruiz.unscale_dual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpresults.y});
	qpwork.ruiz.unscale_dual_in_place_in(
			VectorViewMut<T>{from_eigen, qpresults.z});

	{
		// EigenAllowAlloc _{};
		for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
			qpresults.info.objValue +=
					0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
			qpresults.info.objValue +=
					qpresults.x(j) * T(qpmodel.H.col(j)
			                           .tail(qpmodel.dim - j - 1)
			                           .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
		}
		qpresults.info.objValue += (qpmodel.g).dot(qpresults.x);
	}
}

template <typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using MatRef =
		Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const>;

template <typename Mat, typename T>
void QPsetup_generic( //
		Mat const& H,
		VecRef<T> g,
		Mat const& A,
		VecRef<T> b,
		Mat const& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::Settings<T>& qpsettings,
		qp::dense::Data<T>& qpmodel,
		qp::dense::Workspace<T>& qpwork,
		qp::Results<T>& qpresults) {

	auto start = std::chrono::high_resolution_clock::now();
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

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);

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
	}

	qpwork.rhs.setZero();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	qpresults.info.setup_time = duration.count();
}

template <typename T>
void QPsetup_dense( //
		MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l,
		qp::Settings<T>& qpsettings,
		qp::dense::Data<T>& qpmodel,
		qp::dense::Workspace<T>& qpwork,
		qp::Results<T>& qpresults

) {
	dense::QPsetup_generic(
			H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

template <typename T>
void QPsetup( //
		const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::Settings<T>& qpsettings,
		qp::dense::Data<T>& qpmodel,
		qp::dense::Workspace<T>& qpwork,
		qp::Results<T>& qpresults) {
	dense::QPsetup_generic(
			H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

} // namespace dense

} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_SOLVER_HPP */
