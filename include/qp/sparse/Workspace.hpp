/** \file */

#ifndef PROXSUITE_QP_SPARSE_WORKSPACE_HPP
#define PROXSUITE_QP_SPARSE_WORKSPACE_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/Settings.hpp>
#include <veg/vec.hpp>
#include "qp/sparse/views.hpp"
#include "qp/sparse/Data.hpp"
#include "qp/Results.hpp"

#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {

template <typename T, typename I>
struct Workspace {

	struct /* NOLINT */ {
		veg::Vec<veg::mem::byte> storage;
		veg::Vec<I> ldl_col_ptrs;
		veg::Vec<I> perm_inv;
		bool do_ldlt;

		auto stack_mut() -> veg::dynstack::DynStackMut {
			return {
					veg::from_slice_mut,
					storage.as_mut(),
			};
		}

		void setup_impl(
				QpView<T, I> qp,
				Data<T, I>& data,
				veg::dynstack::StackReq precond_req) {
			data.dim = qp.H.nrows();
			data.n_eq = qp.AT.ncols();
			data.n_in = qp.CT.ncols();
			data.H_nnz = qp.H.nnz();
			data.A_nnz = qp.AT.nnz();
			data.C_nnz = qp.CT.nnz();

			data.g = qp.g.to_eigen();
			data.b = qp.b.to_eigen();
			data.l = qp.l.to_eigen();
			data.u = qp.u.to_eigen();

			using namespace veg::dynstack;
			using namespace linearsolver::sparse::util;

			using SR = StackReq;
			veg::Tag<I> itag;
			veg::Tag<T> xtag;

			isize n = qp.H.nrows();
			isize n_eq = qp.AT.ncols();
			isize n_in = qp.CT.ncols();
			isize n_tot = n + n_eq + n_in;

			isize nnz_tot = qp.H.nnz() + qp.AT.nnz() + qp.CT.nnz();

			// form the full kkt matrix
			// assuming H, AT, CT are sorted
			// and H is upper triangular
			{
				data.kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
				data.kkt_row_indices.resize_for_overwrite(nnz_tot);
				data.kkt_values.resize_for_overwrite(nnz_tot);

				I* kktp = data.kkt_col_ptrs.ptr_mut();
				I* kkti = data.kkt_row_indices.ptr_mut();
				T* kktx = data.kkt_values.ptr_mut();

				kktp[0] = 0;
				usize col = 0;
				usize pos = 0;

				auto insert_submatrix = [&](linearsolver::sparse::MatRef<T, I> m,
				                            bool assert_sym_hi) -> void {
					I const* mi = m.row_indices();
					T const* mx = m.values();
					isize ncols = m.ncols();

					for (usize j = 0; j < usize(ncols); ++j) {
						usize col_start = m.col_start(j);
						usize col_end = m.col_end(j);

						kktp[col + 1] =
								checked_non_negative_plus(kktp[col], I(col_end - col_start));
						++col;

						for (usize p = col_start; p < col_end; ++p) {
							usize i = zero_extend(mi[p]);
							if (assert_sym_hi) {
								VEG_ASSERT(i <= j);
							}

							kkti[pos] = veg::nb::narrow<I>{}(i);
							kktx[pos] = mx[p];

							++pos;
						}
					}
				};

				insert_submatrix(qp.H, true);
				insert_submatrix(qp.AT, false);
				insert_submatrix(qp.CT, false);
			}

			storage.resize_for_overwrite( //
					(StackReq::with_len(itag, n_tot) &
			     linearsolver::sparse::factorize_symbolic_req( //
							 itag,                                     //
							 n_tot,                                    //
							 nnz_tot,                                  //
							 linearsolver::sparse::Ordering::amd))     //
							.alloc_req()                               //
			);

			ldl_col_ptrs.resize_for_overwrite(n_tot + 1);
			perm_inv.resize_for_overwrite(n_tot);

			DynStackMut stack = stack_mut();

			bool overflow = false;
			{
				auto _etree = stack.make_new_for_overwrite(itag, n_tot).unwrap();
				auto etree = _etree.ptr_mut();

				using namespace veg::literals;
				auto kkt_sym = linearsolver::sparse::SymbolicMatRef<I>{
						linearsolver::sparse::from_raw_parts,
						n_tot,
						n_tot,
						nnz_tot,
						data.kkt_col_ptrs.ptr(),
						nullptr,
						data.kkt_row_indices.ptr(),
				};
				linearsolver::sparse::factorize_symbolic_non_zeros( //
						ldl_col_ptrs.ptr_mut() + 1,
						etree,
						perm_inv.ptr_mut(),
						static_cast<I const*>(nullptr),
						kkt_sym,
						stack);

				auto pcol_ptrs = ldl_col_ptrs.ptr_mut();
				pcol_ptrs[0] = I(0);

				using veg::u64;
				u64 acc = 0;

				for (usize i = 0; i < usize(n_tot); ++i) {
					acc += u64(zero_extend(pcol_ptrs[i + 1]));
					if (acc != u64(I(acc))) {
						overflow = true;
					}
					pcol_ptrs[(i + 1)] = I(acc);
				}
			}

			auto lnnz = isize(zero_extend(ldl_col_ptrs[n_tot]));

			// if qp much sparser than kkt
			// do_ldlt = !overflow && lnnz < (100 * nnz_tot);
			do_ldlt = !overflow && lnnz < 10000000;

#define PROX_QP_ALL_OF(...)                                                    \
	::veg::dynstack::StackReq::and_(::veg::init_list(__VA_ARGS__))
#define PROX_QP_ANY_OF(...)                                                    \
	::veg::dynstack::StackReq::or_(::veg::init_list(__VA_ARGS__))

			auto refactorize_req =
					do_ldlt
							? PROX_QP_ANY_OF({
										linearsolver::sparse::
												factorize_symbolic_req( // symbolic ldl
														itag,
														n_tot,
														nnz_tot,
														linearsolver::sparse::Ordering::user_provided),
										PROX_QP_ALL_OF({
												SR::with_len(xtag, n_tot), // diag
												linearsolver::sparse::
														factorize_numeric_req( // numeric ldl
																xtag,
																itag,
																n_tot,
																nnz_tot,
																linearsolver::sparse::Ordering::user_provided),
										}),
								})
							: PROX_QP_ALL_OF({
										SR::with_len(itag, 0),
										SR::with_len(xtag, 0),
								});

			auto x_vec = [&](isize n) noexcept -> StackReq {
				return linearsolver::dense::temp_vec_req(xtag, n);
			};

			auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // tmp
					x_vec(n_tot), // err
					x_vec(n_tot), // work
			});

			auto unscaled_primal_dual_residual_req = x_vec(n); // Hx
			auto line_search_req = PROX_QP_ALL_OF({
					x_vec(2 * n_in), // alphas
					x_vec(n),        // Cdx_active
					x_vec(n_in),     // active_part_z
					x_vec(n_in),     // tmp_lo
					x_vec(n_in),     // tmp_up
			});
			auto primal_dual_newton_semi_smooth_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // dw
					PROX_QP_ANY_OF({
							ldl_solve_in_place_req,
							PROX_QP_ALL_OF({
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_lo
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_up
									SR::with_len(
											veg::Tag<bool>{}, n_in), // new_active_constraints
									(do_ldlt && n_in > 0)
											? PROX_QP_ANY_OF({
														linearsolver::sparse::add_row_req(
																xtag, itag, n_tot, false, n, n_tot),
														linearsolver::sparse::delete_row_req(
																xtag, itag, n_tot, n_tot),
												})
											: refactorize_req,
							}),
							PROX_QP_ALL_OF({
									x_vec(n),    // Hdx
									x_vec(n_eq), // Adx
									x_vec(n_in), // Cdx
									x_vec(n),    // ATdy
									x_vec(n),    // CTdz
							}),
					}),
					line_search_req,
			});

			auto iter_req = PROX_QP_ANY_OF({
					PROX_QP_ALL_OF(
							{x_vec(n_eq), // primal_residual_eq_scaled
			         x_vec(n_in), // primal_residual_in_scaled_lo
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n),    // dual_residual_scaled
			         PROX_QP_ANY_OF({
									 unscaled_primal_dual_residual_req,
									 PROX_QP_ALL_OF({
											 x_vec(n),    // x_prev
											 x_vec(n_eq), // y_prev
											 x_vec(n_in), // z_prev
											 primal_dual_newton_semi_smooth_req,
									 }),
							 })}),
					refactorize_req, // mu_update
			});

			auto req = //
					PROX_QP_ALL_OF({
							x_vec(n),                             // g_scaled
							x_vec(n_eq),                          // b_scaled
							x_vec(n_in),                          // l_scaled
							x_vec(n_in),                          // u_scaled
							SR::with_len(veg::Tag<bool>{}, n_in), // active constr
							SR::with_len(itag, n_tot),            // kkt nnz counts
							refactorize_req,
							PROX_QP_ANY_OF({
									precond_req,
									PROX_QP_ALL_OF({
											do_ldlt ? PROX_QP_ALL_OF({
																		SR::with_len(itag, n_tot), // perm
																		SR::with_len(itag, n_tot), // etree
																		SR::with_len(itag, n_tot), // ldl nnz counts
																		SR::with_len(itag, lnnz), // ldl row indices
																		SR::with_len(xtag, lnnz), // ldl values
																})
															: PROX_QP_ALL_OF({
																		SR::with_len(itag, 0),
																		SR::with_len(xtag, 0),
																}),
											iter_req,
									}),
							}),
					});

			storage.resize_for_overwrite(req.alloc_req());
		}
	} _;

	Workspace() = default;

	auto ldl_col_ptrs() const -> I const* {
		return _.ldl_col_ptrs.ptr();
	}
	auto ldl_col_ptrs_mut() -> I* {
		return _.ldl_col_ptrs.ptr_mut();
	}
	auto stack_mut() -> veg::dynstack::DynStackMut {
		return _.stack_mut();
	}
};

}//namespace sparse
}//namespace qp
}//namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WORKSPACE_HPP */