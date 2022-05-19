/** \file */

#ifndef PROXSUITE_QP_SPARSE_SOLVER_HPP
#define PROXSUITE_QP_SPARSE_SOLVER_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/settings.hpp>
#include <veg/vec.hpp>
#include "qp/results.hpp"
#include "qp/sparse/views.hpp"
#include "qp/sparse/model.hpp"
#include "qp/sparse/workspace.hpp"
#include "qp/sparse/utils.hpp"
#include "qp/sparse/preconditioner/ruiz.hpp"
#include "qp/sparse/preconditioner/identity.hpp"

#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {
using veg::isize;
using veg::usize;
using veg::i64;
using dense::infty_norm;

template <typename T, typename I>
void ldl_solve(
		VectorViewMut<T> sol,
		VectorView<T> rhs,
		isize n_tot,
		linearsolver::sparse::MatMut<T, I> ldl,
		Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>& iterative_solver,
		bool do_ldlt,
		veg::dynstack::DynStackMut stack,
		T* ldl_values,
		I* perm,
		I* ldl_col_ptrs,
		I const* perm_inv) {
	LDLT_TEMP_VEC_UNINIT(T, work_, n_tot, stack);
	auto rhs_e = rhs.to_eigen();
	auto sol_e = sol.to_eigen();
	auto zx = linearsolver::sparse::util::zero_extend;

	if (do_ldlt) {

		for (isize i = 0; i < n_tot; ++i) {
			work_[i] = rhs_e[isize(zx(perm[i]))];
		}

		linearsolver::sparse::dense_lsolve<T, I>( //
				{linearsolver::sparse::from_eigen, work_},
				ldl.as_const());

		for (isize i = 0; i < n_tot; ++i) {
			work_[i] /= ldl_values[isize(zx(ldl_col_ptrs[i]))];
		}

		linearsolver::sparse::dense_ltsolve<T, I>( //
				{linearsolver::sparse::from_eigen, work_},
				ldl.as_const());

		for (isize i = 0; i < n_tot; ++i) {
			sol_e[i] = work_[isize(zx(perm_inv[i]))];
		}
	} else {
		work_ = iterative_solver.solve(rhs_e);
		sol_e = work_;
	}
};

template <typename T, typename I>
void ldl_iter_solve_noalias(
		VectorViewMut<T> sol,
		VectorView<T> rhs,
		VectorView<T> init_guess,
		Results<T> const& results,
		Model<T, I> const& data,
		isize n_tot,
		linearsolver::sparse::MatMut<T, I> ldl,
		Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>& iterative_solver,
		bool do_ldlt,
		veg::dynstack::DynStackMut stack,
		T* ldl_values,
		I* perm,
		I* ldl_col_ptrs,
		I const* perm_inv,
		Settings<T> const& settings,
		linearsolver::sparse::MatMut<T, I> kkt_active,
		veg::SliceMut<bool> active_constraints) {
	auto rhs_e = rhs.to_eigen();
	auto sol_e = sol.to_eigen();

	if (init_guess.dim == sol.dim) {
		sol_e = init_guess.to_eigen();
	} else {
		sol_e.setZero();
	}

	LDLT_TEMP_VEC_UNINIT(T, err, n_tot, stack);

	T prev_err_norm = std::numeric_limits<T>::infinity();

	for (isize solve_iter = 0; solve_iter < settings.nb_iterative_refinement;
	     ++solve_iter) {

		auto err_x = err.head(data.dim);
		auto err_y = err.segment(data.dim, data.n_eq);
		auto err_z = err.tail(data.n_in);

		auto sol_x = sol_e.head(data.dim);
		auto sol_y = sol_e.segment(data.dim, data.n_eq);
		auto sol_z = sol_e.tail(data.n_in); // removed active set condition

		err = -rhs_e;

		if (solve_iter > 0) {
			T mu_eq_neg = -results.info.mu_eq;
			T mu_in_neg = -results.info.mu_in;
			detail::noalias_symhiv_add(err, kkt_active.to_eigen(), sol_e);
			err_x += results.info.rho * sol_x;
			err_y += mu_eq_neg * sol_y;
			for (isize i = 0; i < data.n_in; ++i) {
				err_z[i] += (active_constraints[i] ? mu_in_neg : T(1)) * sol_z[i];
			}
		}

		T err_norm = infty_norm(err);
		if (err_norm > prev_err_norm / T(2)) {
			break;
		}
		prev_err_norm = err_norm;

		ldl_solve(
				{qp::from_eigen, err},
				{qp::from_eigen, err},
				n_tot,
				ldl,
				iterative_solver,
				do_ldlt,
				stack,
				ldl_values,
				perm,
				ldl_col_ptrs,
				perm_inv);

		sol_e -= err;
	}
};

template <typename T, typename I>
void ldl_solve_in_place(
		VectorViewMut<T> rhs,
		VectorView<T> init_guess,
		Results<T> const& results,
		Model<T, I> const& data,
		isize n_tot,
		linearsolver::sparse::MatMut<T, I> ldl,
		Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>& iterative_solver,
		bool do_ldlt,
		veg::dynstack::DynStackMut stack,
		T* ldl_values,
		I* perm,
		I* ldl_col_ptrs,
		I const* perm_inv,
		Settings<T> const& settings,
		linearsolver::sparse::MatMut<T, I> kkt_active,
		veg::SliceMut<bool> active_constraints) {
	LDLT_TEMP_VEC_UNINIT(T, tmp, n_tot, stack);
	ldl_iter_solve_noalias(
			{qp::from_eigen, tmp},
			rhs.as_const(),
			init_guess,
			results,
			data,
			n_tot,
			ldl,
			iterative_solver,
			do_ldlt,
			stack,
			ldl_values,
			perm,
			ldl_col_ptrs,
			perm_inv,
			settings,
			kkt_active,
			active_constraints);
	rhs.to_eigen() = tmp;
};

template <typename T>
using DMat = Eigen::Matrix<T, -1, -1>;

template <typename T, typename I>
auto inner_reconstructed_matrix(
		linearsolver::sparse::MatMut<T, I> ldl, bool do_ldlt) -> DMat<T> {
	VEG_ASSERT(do_ldlt);
	auto ldl_dense = ldl.to_eigen().toDense();
	auto l = DMat<T>(ldl_dense.template triangularView<Eigen::UnitLower>());
	auto lt = l.transpose();
	auto d = ldl_dense.diagonal().asDiagonal();
	auto mat = DMat<T>(l * d * lt);
	return mat;
};

template <typename T, typename I>
auto reconstructed_matrix(
		linearsolver::sparse::MatMut<T, I> ldl,
		bool do_ldlt,
		I const* perm_inv,
		isize n_tot) -> DMat<T> {
	auto mat = inner_reconstructed_matrix(ldl, do_ldlt);
	auto mat_backup = mat;
	for (isize i = 0; i < n_tot; ++i) {
		for (isize j = 0; j < n_tot; ++j) {
			mat(i, j) = mat_backup(perm_inv[i], perm_inv[j]);
		}
	}
	return mat;
};

template <typename T, typename I>
auto reconstruction_error(
		linearsolver::sparse::MatMut<T, I> ldl,
		bool do_ldlt,
		I const* perm_inv,
		Results<T> const& results,
		Model<T, I> const& data,
		isize n_tot,
		linearsolver::sparse::MatMut<T, I> kkt_active,
		veg::SliceMut<bool> active_constraints) -> DMat<T> {
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	auto diff = DMat<T>(
			reconstructed_matrix(ldl, do_ldlt, perm_inv, n_tot) -
			DMat<T>(DMat<T>(kkt_active.to_eigen())
	                .template selfadjointView<Eigen::Upper>()));
	diff.diagonal().head(data.dim).array() -= results.info.rho;
	diff.diagonal().segment(data.dim, data.n_eq).array() -= mu_eq_neg;
	for (isize i = 0; i < data.n_in; ++i) {
		diff.diagonal()[data.dim + data.n_eq + i] -=
				active_constraints[i] ? mu_in_neg : T(1);
	}
	return diff;
};

template <typename T>
struct PrimalDualGradResult {
	T a;
	T b;
	T grad;
	VEG_REFLECT(PrimalDualGradResult, a, b, grad);
};

template <typename T, typename I, typename P>
void qp_solve(
		Results<T>& results,
		Model<T, I>& data,
		Settings<T> const& settings,
		Workspace<T, I>& work,
		P& precond) {

	using namespace veg::literals;
	namespace util = linearsolver::sparse::util;
	auto zx = util::zero_extend;

	veg::dynstack::DynStackMut stack = work.stack_mut();

	isize n = data.dim;
	isize n_eq = data.n_eq;
	isize n_in = data.n_in;
	isize n_tot = n + n_eq + n_in;

	VectorViewMut<T> x{qp::from_eigen, results.x};
	VectorViewMut<T> y{qp::from_eigen, results.y};
	VectorViewMut<T> z{qp::from_eigen, results.z};

	linearsolver::sparse::MatMut<T, I> kkt = data.kkt_mut();

	auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n);

	linearsolver::sparse::MatMut<T, I> H_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);

	linearsolver::sparse::MatMut<T, I> AT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, data.A_nnz);

	linearsolver::sparse::MatMut<T, I> CT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, data.C_nnz);

	auto& g_scaled_e = work._.g_scaled;
	auto& b_scaled_e = work._.b_scaled;
	auto& l_scaled_e = work._.l_scaled;
	auto& u_scaled_e = work._.u_scaled;

	QpViewMut<T, I> qp_scaled = {
			H_scaled,
			{linearsolver::sparse::from_eigen, g_scaled_e},
			AT_scaled,
			{linearsolver::sparse::from_eigen, b_scaled_e},
			CT_scaled,
			{linearsolver::sparse::from_eigen, l_scaled_e},
			{linearsolver::sparse::from_eigen, u_scaled_e},
	};

	T const primal_feasibility_rhs_1_eq = infty_norm(data.b);
	T const primal_feasibility_rhs_1_in_u = infty_norm(data.u);
	T const primal_feasibility_rhs_1_in_l = infty_norm(data.l);
	T const dual_feasibility_rhs_2 = infty_norm(data.g);

	auto ldl_col_ptrs = work.ldl_col_ptrs_mut();

	veg::Tag<I> itag;
	veg::Tag<T> xtag;

	bool do_ldlt = work._.do_ldlt;

	isize ldlt_ntot = do_ldlt ? n_tot : 0;

	auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();

	I* perm_inv = work._.perm_inv.ptr_mut();
	I* perm = _perm.ptr_mut();

	if (do_ldlt) {
		// compute perm from perm_inv
		for (isize i = 0; i < n_tot; ++i) {
			perm[isize(zx(perm_inv[i]))] = I(i);
		}
	}

	I* kkt_nnz_counts = work._.kkt_nnz_counts.ptr_mut();

	auto& iterative_solver = *work._.matrix_free_solver.get();

	linearsolver::sparse::MatMut<T, I> kkt_active = {
			linearsolver::sparse::from_raw_parts,
			n_tot,
			n_tot,
			data.H_nnz + data.A_nnz,
			kkt.col_ptrs_mut(),
			kkt_nnz_counts,
			kkt.row_indices_mut(),
			kkt.values_mut(),
	};

	I* etree = work._.etree.ptr_mut();
	I* ldl_nnz_counts = work._.ldl_nnz_counts.ptr_mut();
	I* ldl_row_indices = work._.ldl_row_indices.ptr_mut();
	T* ldl_values = work._.ldl_values.ptr_mut();
	veg::SliceMut<bool> active_constraints = results.active_constraints.as_mut();

	linearsolver::sparse::MatMut<T, I> ldl = {
			linearsolver::sparse::from_raw_parts,
			n_tot,
			n_tot,
			0,
			ldl_col_ptrs,
			do_ldlt ? ldl_nnz_counts : nullptr,
			ldl_row_indices,
			ldl_values,
	};

	auto& aug_kkt = *work._.matrix_free_kkt.get();

	T bcl_eta_ext_init = pow(T(0.1), settings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(settings.eps_abs, T(1e-9));

	auto x_e = x.to_eigen();
	auto y_e = y.to_eigen();
	auto z_e = z.to_eigen();

	if (!settings.warm_start) {
		LDLT_TEMP_VEC_UNINIT(T, rhs, n_tot, stack);
		LDLT_TEMP_VEC_UNINIT(T, no_guess, 0, stack);

		rhs.head(n) = -g_scaled_e;
		rhs.segment(n, n_eq) = b_scaled_e;
		rhs.segment(n + n_eq, n_in).setZero();

		ldl_solve_in_place(
				{qp::from_eigen, rhs},
				{qp::from_eigen, no_guess},
				results,
				data,
				n_tot,
				ldl,
				iterative_solver,
				do_ldlt,
				stack,
				ldl_values,
				perm,
				ldl_col_ptrs,
				perm_inv,
				settings,
				kkt_active,
				active_constraints);
		x_e = rhs.head(n);
		y_e = rhs.segment(n, n_eq);
		z_e = rhs.segment(n + n_eq, n_in);
	}

	for (isize iter = 0; iter < settings.max_iter; ++iter) {

		results.info.iter_ext += 1;
		if (iter == settings.max_iter) {
			break;
		}
		T new_bcl_mu_eq = results.info.mu_eq;
		T new_bcl_mu_in = results.info.mu_in;
		T new_bcl_mu_eq_inv = results.info.mu_eq_inv;
		T new_bcl_mu_in_inv = results.info.mu_in_inv;

		{
			T primal_feasibility_eq_rhs_0;
			T primal_feasibility_in_rhs_0;

			T dual_feasibility_rhs_0(0);
			T dual_feasibility_rhs_1(0);
			T dual_feasibility_rhs_3(0);

			LDLT_TEMP_VEC_UNINIT(T, primal_residual_eq_scaled, n_eq, stack);
			LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_lo, n_in, stack);
			LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_up, n_in, stack);

			LDLT_TEMP_VEC_UNINIT(T, dual_residual_scaled, n, stack);

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto is_primal_feasible = [&](T primal_feasibility_lhs) -> bool {
				T rhs_pri = settings.eps_abs;
				if (settings.eps_rel != 0) {
					rhs_pri += settings.eps_rel * std::max({
																						primal_feasibility_eq_rhs_0,
																						primal_feasibility_in_rhs_0,
																						primal_feasibility_rhs_1_eq,
																						primal_feasibility_rhs_1_in_l,
																						primal_feasibility_rhs_1_in_u,
																				});
				}
				return primal_feasibility_lhs <= rhs_pri;
			};
			auto is_dual_feasible = [&](T dual_feasibility_lhs) -> bool {
				T rhs_dua = settings.eps_abs;
				if (settings.eps_rel != 0) {
					rhs_dua += settings.eps_rel * std::max({
																						dual_feasibility_rhs_0,
																						dual_feasibility_rhs_1,
																						dual_feasibility_rhs_2,
																						dual_feasibility_rhs_3,
																				});
				}

				return dual_feasibility_lhs <= rhs_dua;
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			VEG_BIND(
					auto,
					(primal_feasibility_lhs, dual_feasibility_lhs),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							data,
							qp_scaled.as_const(),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));

			if (settings.verbose) {
				std::cout << "-------- outer iteration: " << iter << " primal residual "
									<< primal_feasibility_lhs << " dual residual "
									<< dual_feasibility_lhs << " mu_in " << results.info.mu_in
									<< " bcl_eta_ext " << bcl_eta_ext << " bcl_eta_in "
									<< bcl_eta_in << std::endl;
			}
			if (is_primal_feasible(primal_feasibility_lhs) &&
			    is_dual_feasible(dual_feasibility_lhs)) {
				break;
			}

			LDLT_TEMP_VEC_UNINIT(T, x_prev_e, n, stack);
			LDLT_TEMP_VEC_UNINIT(T, y_prev_e, n_eq, stack);
			LDLT_TEMP_VEC_UNINIT(T, z_prev_e, n_in, stack);
			LDLT_TEMP_VEC(T, dw_prev, n_tot, stack);

			x_prev_e = x_e;
			y_prev_e = y_e;
			z_prev_e = z_e;

			// Cx + 1/mu_in * z_prev
			primal_residual_in_scaled_up += results.info.mu_in * z_prev_e;
			primal_residual_in_scaled_lo = primal_residual_in_scaled_up;

			// Cx - l + 1/mu_in * z_prev
			primal_residual_in_scaled_lo -= l_scaled_e;

			// Cx - u + 1/mu_in * z_prev
			primal_residual_in_scaled_up -= u_scaled_e;

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto primal_dual_newton_semi_smooth = [&]() -> void {
				for (isize iter_inner = 0; iter_inner < settings.max_iter_in;
				     ++iter_inner) {
					LDLT_TEMP_VEC_UNINIT(T, dw, n_tot, stack);

					if (iter_inner == settings.max_iter_in - 1) {
						results.info.iter += settings.max_iter_in;
						break;
					}

					if (settings.verbose) {
						std::cout
								<< "-------------------starting inner loop solve in place "
								<< std::endl;
					}
					// primal_dual_semi_smooth_newton_step
					{
						LDLT_TEMP_VEC_UNINIT(bool, active_set_lo, n_in, stack);
						LDLT_TEMP_VEC_UNINIT(bool, active_set_up, n_in, stack);
						LDLT_TEMP_VEC_UNINIT(bool, new_active_constraints, n_in, stack);
						auto rhs = dw;

						active_set_lo.array() = primal_residual_in_scaled_lo.array() <= 0;
						active_set_up.array() = primal_residual_in_scaled_up.array() >= 0;
						new_active_constraints = active_set_lo || active_set_up;

						// active set change
						if (n_in > 0) {
							bool removed = false;
							bool added = false;
							veg::unused(removed, added);

							for (isize i = 0; i < n_in; ++i) {
								bool was_active = active_constraints[i];
								bool is_active = new_active_constraints[i];

								isize idx = n + n_eq + i;

								usize col_nnz =
										zx(kkt.col_end(usize(idx))) - zx(kkt.col_start(usize(idx)));

								if (is_active && !was_active) {
									added = true;

									kkt_active.nnz_per_col_mut()[idx] = I(col_nnz);
									kkt_active._set_nnz(kkt_active.nnz() + isize(col_nnz));

									if (do_ldlt) {
										linearsolver::sparse::VecRef<T, I> new_col{
												linearsolver::sparse::from_raw_parts,
												n_tot,
												isize(col_nnz),
												kkt.row_indices() + zx(kkt.col_start(usize(idx))),
												kkt.values() + zx(kkt.col_start(usize(idx))),
										};

										ldl = linearsolver::sparse::add_row(
												ldl,
												etree,
												perm_inv,
												idx,
												new_col,
												-results.info.mu_in,
												stack);
									}
									active_constraints[i] = new_active_constraints[i];

								} else if (!is_active && was_active) {
									removed = true;
									kkt_active.nnz_per_col_mut()[idx] = 0;
									kkt_active._set_nnz(kkt_active.nnz() - isize(col_nnz));
									if (do_ldlt) {
										ldl = linearsolver::sparse::delete_row(
												ldl, etree, perm_inv, idx, stack);
									}
									active_constraints[i] = new_active_constraints[i];
								}
							}

							if (!do_ldlt) {
								if (removed || added) {
									refactorize(
											results,
											do_ldlt,
											n_tot,
											kkt_active,
											active_constraints,
											iterative_solver,
											data,
											etree,
											ldl_nnz_counts,
											ldl_row_indices,
											perm_inv,
											ldl_values,
											perm,
											ldl_col_ptrs,
											stack,
											ldl,
											aug_kkt,
											xtag);
								}
							}
						}

						rhs.head(n) = -dual_residual_scaled;
						rhs.segment(n, n_eq) = -primal_residual_eq_scaled;

						for (isize i = 0; i < n_in; ++i) {
							if (active_set_up(i)) {
								rhs(n + n_eq + i) = results.info.mu_in * z_e(i) -
								                    primal_residual_in_scaled_up(i);
							} else if (active_set_lo(i)) {
								rhs(n + n_eq + i) = results.info.mu_in * z_e(i) -
								                    primal_residual_in_scaled_lo(i);
							} else {
								rhs(n + n_eq + i) = -z_e(i);
								rhs.head(n) += z_e(i) * CT_scaled.to_eigen().col(i);
							}
						}

						ldl_solve_in_place(
								{qp::from_eigen, rhs},
								{qp::from_eigen, dw_prev},
								results,
								data,
								n_tot,
								ldl,
								iterative_solver,
								do_ldlt,
								stack,
								ldl_values,
								perm,
								ldl_col_ptrs,
								perm_inv,
								settings,
								kkt_active,
								active_constraints);
					}
					if (settings.verbose) {
						std::cout
								<< "-------------------finished inner loop solve in place "
								<< std::endl;
					}
					auto dx = dw.head(n);
					auto dy = dw.segment(n, n_eq);
					auto dz = dw.segment(n + n_eq, n_in);

					LDLT_TEMP_VEC(T, Hdx, n, stack);
					LDLT_TEMP_VEC(T, Adx, n_eq, stack);
					LDLT_TEMP_VEC(T, Cdx, n_in, stack);

					LDLT_TEMP_VEC(T, ATdy, n, stack);
					LDLT_TEMP_VEC(T, CTdz, n, stack);

					detail::noalias_symhiv_add(Hdx, H_scaled.to_eigen(), dx);
					detail::noalias_gevmmv_add(Adx, ATdy, AT_scaled.to_eigen(), dx, dy);
					detail::noalias_gevmmv_add(Cdx, CTdz, CT_scaled.to_eigen(), dx, dz);

					T alpha = 1;
					// primal dual line search
					if (settings.verbose) {
						std::cout << "-------------------starting inner loop line search "
											<< std::endl;
					}
					if (n_in > 0) {
						auto primal_dual_gradient_norm =
								[&](T alpha_cur) -> PrimalDualGradResult<T> {
							LDLT_TEMP_VEC_UNINIT(T, Cdx_active, n_in, stack);
							LDLT_TEMP_VEC_UNINIT(T, active_part_z, n_in, stack);
							{
								LDLT_TEMP_VEC_UNINIT(T, tmp_lo, n_in, stack);
								LDLT_TEMP_VEC_UNINIT(T, tmp_up, n_in, stack);

								auto zero = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_in);

								tmp_lo = primal_residual_in_scaled_lo + alpha_cur * Cdx;
								tmp_up = primal_residual_in_scaled_up + alpha_cur * Cdx;
								Cdx_active = (tmp_lo.array() < 0 || tmp_up.array() > 0)
								                 .select(Cdx, zero);
								active_part_z =
										(tmp_lo.array() < 0)
												.select(primal_residual_in_scaled_lo, zero) +
										(tmp_up.array() > 0)
												.select(primal_residual_in_scaled_up, zero);
							}

							T a =
									dx.dot(Hdx) +                         //
									results.info.rho * dx.squaredNorm() + //
									results.info.mu_eq_inv * Adx.squaredNorm() +
									+results.info.mu_in_inv * Cdx_active.squaredNorm() +
									results.info.nu * results.info.mu_eq *
											(results.info.mu_eq_inv * Adx - dy).squaredNorm() +
									results.info.nu * results.info.mu_in *
											(results.info.mu_in_inv * Cdx_active - dz).squaredNorm();

							T b =
									x_e.dot(Hdx) + //
									(results.info.rho * (x_e - x_prev_e) + g_scaled_e)
											.dot(dx) + //
									Adx.dot(
											results.info.mu_eq_inv * primal_residual_eq_scaled +
											y_e) +                                               //
									results.info.mu_in_inv * Cdx_active.dot(active_part_z) + //
									results.info.nu * primal_residual_eq_scaled.dot(
																				results.info.mu_eq_inv * Adx - dy) + //
									results.info.nu *
											(active_part_z - results.info.mu_in * z_e)
													.dot(results.info.mu_in_inv * Cdx_active - dz);

							return {
									a,
									b,
									a * alpha_cur + b,
							};
						};

						LDLT_TEMP_VEC_UNINIT(T, alphas, 2 * n_in, stack);
						isize alphas_count = 0;

						for (isize i = 0; i < n_in; ++i) {
							T alpha_candidates[2] = {
									-primal_residual_in_scaled_lo(i) / (Cdx(i)),
									-primal_residual_in_scaled_up(i) / (Cdx(i)),
							};

							for (auto alpha_candidate : alpha_candidates) {
								if (alpha_candidate > 0) {
									alphas[alphas_count] = alpha_candidate;
									++alphas_count;
								}
							}
						}
						std::sort(alphas.data(), alphas.data() + alphas_count);
						alphas_count =
								std::unique(alphas.data(), alphas.data() + alphas_count) -
								alphas.data();

						if (alphas_count > 0 && alphas[0] <= 1) {
							auto infty = std::numeric_limits<T>::infinity();

							T last_neg_grad = 0;
							T alpha_last_neg = 0;
							T first_pos_grad = 0;
							T alpha_first_pos = infty;

							{
								for (isize i = 0; i < alphas_count; ++i) {
									T alpha_cur = alphas[i];
									T gr = primal_dual_gradient_norm(alpha_cur).grad;

									if (gr < 0) {
										alpha_last_neg = alpha_cur;
										last_neg_grad = gr;
									} else {
										first_pos_grad = gr;
										alpha_first_pos = alpha_cur;
										break;
									}
								}

								if (alpha_last_neg == 0) {
									last_neg_grad =
											primal_dual_gradient_norm(alpha_last_neg).grad;
								}

								if (alpha_first_pos == infty) {
									auto res = primal_dual_gradient_norm(2 * alpha_last_neg + 1);
									alpha = -res.b / res.a;
								} else {
									alpha = alpha_last_neg -
									        last_neg_grad * (alpha_first_pos - alpha_last_neg) /
									            (first_pos_grad - last_neg_grad);
								}
							}
						} else {
							auto res = primal_dual_gradient_norm(T(0));
							alpha = -res.b / res.a;
						}
					}
					if (alpha * infty_norm(dw) < T(1e-11) && iter_inner > 0) {
						results.info.iter += iter_inner + 1;
						return;
					}
					if (settings.verbose) {
						std::cout << "-------------------finished inner loop line search "
											<< std::endl;
					}

					x_e += alpha * dx;
					y_e += alpha * dy;
					z_e += alpha * dz;

					dual_residual_scaled +=
							alpha * (Hdx + ATdy + CTdz + results.info.rho * dx);
					primal_residual_eq_scaled += alpha * (Adx - results.info.mu_eq * dy);
					primal_residual_in_scaled_lo += alpha * Cdx;
					primal_residual_in_scaled_up += alpha * Cdx;

					T err_in = std::max({
							(infty_norm(
									detail::negative_part(primal_residual_in_scaled_lo) +
									detail::positive_part(primal_residual_in_scaled_up) -
									results.info.mu_in * z_e)),
							(infty_norm(primal_residual_eq_scaled)),
							(infty_norm(dual_residual_scaled)),
					});
					if (settings.verbose) {
						std::cout << "--inner iter " << iter_inner << " iner error "
											<< err_in << " alpha " << alpha << " infty_norm(dw) "
											<< infty_norm(dw) << std::endl;
					}
					if (err_in <= bcl_eta_in) {
						results.info.iter += iter_inner + 1;
						return;
					}
				}
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			primal_dual_newton_semi_smooth();

			VEG_BIND(
					auto,
					(primal_feasibility_lhs_new, dual_feasibility_lhs_new),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							data,
							qp_scaled.as_const(),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));
			if (is_primal_feasible(primal_feasibility_lhs_new) &&
			    is_dual_feasible(dual_feasibility_lhs_new)) {
				break;
			}

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto bcl_update = [&]() -> void {
				if (primal_feasibility_lhs_new <= bcl_eta_ext) {
					bcl_eta_ext *= pow(results.info.mu_in, settings.beta_bcl);
					bcl_eta_in = std::max(bcl_eta_in * results.info.mu_in, eps_in_min);

				} else {
					y_e = y_prev_e;
					z_e = z_prev_e;
					new_bcl_mu_in = std::max(
							results.info.mu_in * settings.mu_update_factor,
							settings.mu_max_in);
					new_bcl_mu_eq = std::max(
							results.info.mu_eq * settings.mu_update_factor,
							settings.mu_max_eq);

					new_bcl_mu_in_inv = std::min(
							results.info.mu_in_inv * settings.mu_update_inv_factor,
							settings.mu_max_in_inv);
					new_bcl_mu_eq_inv = std::min(
							results.info.mu_eq_inv * settings.mu_update_inv_factor,
							settings.mu_max_eq_inv);
					bcl_eta_ext =
							bcl_eta_ext_init * pow(new_bcl_mu_in, settings.alpha_bcl);
					bcl_eta_in = std::max(new_bcl_mu_in, eps_in_min);
				}
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			bcl_update();

			VEG_BIND(
					auto,
					(_, dual_feasibility_lhs_new_2),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							data,
							qp_scaled.as_const(),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));
			veg::unused(_);

			if (primal_feasibility_lhs_new >= primal_feasibility_lhs && //
			    dual_feasibility_lhs_new_2 >= primal_feasibility_lhs && //
			    results.info.mu_in <= T(1.E-5)) {
				new_bcl_mu_in = settings.cold_reset_mu_in;
				new_bcl_mu_eq = settings.cold_reset_mu_eq;
				new_bcl_mu_in_inv = settings.cold_reset_mu_in_inv;
				new_bcl_mu_eq_inv = settings.cold_reset_mu_eq_inv;
			}
		}
		if (results.info.mu_in != new_bcl_mu_in ||
		    results.info.mu_eq != new_bcl_mu_eq) {
			{ ++results.info.mu_updates; }
			refactorize(
					results,
					do_ldlt,
					n_tot,
					kkt_active,
					active_constraints,
					iterative_solver,
					data,
					etree,
					ldl_nnz_counts,
					ldl_row_indices,
					perm_inv,
					ldl_values,
					perm,
					ldl_col_ptrs,
					stack,
					ldl,
					aug_kkt,
					xtag);
		}

		results.info.mu_eq = new_bcl_mu_eq;
		results.info.mu_in = new_bcl_mu_in;
		results.info.mu_eq_inv = new_bcl_mu_eq_inv;
		results.info.mu_in_inv = new_bcl_mu_in_inv;
	}

	LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
	tmp.setZero();
	detail::noalias_symhiv_add(tmp, qp_scaled.H.to_eigen(), x_e);
	precond.unscale_dual_residual_in_place({qp::from_eigen, tmp});

	precond.unscale_primal_in_place({qp::from_eigen, x_e});
	precond.unscale_dual_in_place_eq({qp::from_eigen, y_e});
	precond.unscale_dual_in_place_in({qp::from_eigen, z_e});
	tmp *= 0.5;
	tmp += data.g;
	results.info.objValue = (tmp).dot(x_e);
}
} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_SOLVER_HPP */
