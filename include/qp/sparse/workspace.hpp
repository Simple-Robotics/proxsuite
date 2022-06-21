/** \file */

#ifndef PROXSUITE_QP_SPARSE_WORKSPACE_HPP
#define PROXSUITE_QP_SPARSE_WORKSPACE_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/timings.hpp>
#include <qp/settings.hpp>
#include <qp/dense/views.hpp>
#include <veg/vec.hpp>
#include "qp/sparse/views.hpp"
#include "qp/sparse/model.hpp"
#include "qp/results.hpp"
#include "qp/sparse/utils.hpp"

#include <iostream>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {
/*!
 * Workspace class of the sparse solver.
*/
template <typename T, typename I>
struct Workspace;

template <typename T, typename I>
void refactorize(
		Workspace<T,I>& work,
		Results<T> const& results,
		linearsolver::sparse::MatMut<T, I> kkt_active,
		veg::SliceMut<bool> active_constraints,
		Model<T, I> const& data,
		veg::dynstack::DynStackMut stack,
		veg::Tag<T>& xtag) {
		isize n_tot = kkt_active.nrows();
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	
	if (work.internal.do_ldlt) {
		linearsolver::sparse::factorize_symbolic_non_zeros(
				work.internal.ldl.nnz_counts.ptr_mut(), work.internal.ldl.etree.ptr_mut(), work.internal.ldl.perm_inv.ptr_mut(), work.internal.ldl.perm.ptr_mut(), kkt_active.symbolic(), stack);

		isize nnz = 0;
		for (usize j =  0;j<usize(kkt_active.ncols());++j){
			nnz+=usize(kkt_active.col_end(j)-kkt_active.col_start(j));
		}
		VEG_ASSERT(kkt_active.nnz()==nnz);
		auto _diag = stack.make_new_for_overwrite(xtag, n_tot);
		T* diag = _diag.ptr_mut();

		for (isize i = 0; i < data.dim; ++i) {
			diag[i] = results.info.rho;
		}
		for (isize i = 0; i < data.n_eq; ++i) {
			diag[data.dim + i] = mu_eq_neg;
		}
		for (isize i = 0; i < data.n_in; ++i) {
			diag[(data.dim + data.n_eq) + i] =
					active_constraints[i] ? mu_in_neg : T(1);
		}

		linearsolver::sparse::factorize_numeric(
				work.internal.ldl.values.ptr_mut(),
				work.internal.ldl.row_indices.ptr_mut(),
				diag,
				work.internal.ldl.perm.ptr_mut(),
				work.internal.ldl.col_ptrs.ptr(),
				work.internal.ldl.etree.ptr_mut(),
				work.internal.ldl.perm_inv.ptr_mut(),
				kkt_active.as_const(),
				stack);
	} else {
		*work.internal.matrix_free_kkt = {
				{kkt_active.as_const(),
		     active_constraints.as_const(),
		     data.dim,
		     data.n_eq,
		     data.n_in,
		     results.info.rho,
		     results.info.mu_eq_inv,
		     results.info.mu_in_inv}};
		(*work.internal.matrix_free_solver).compute(*work.internal.matrix_free_kkt);
	}
};

template <typename T, typename I>
struct Ldlt {
	veg::Vec<I> etree;
	veg::Vec<I> perm;
	veg::Vec<I> perm_inv;
	veg::Vec<I> col_ptrs;
	veg::Vec<I> nnz_counts;
	veg::Vec<I> row_indices;
	veg::Vec<T> values;
};
template <typename T, typename I>
struct Workspace {

	struct /* NOLINT */ {
		// temporary allocations
		veg::Vec<veg::mem::byte> storage;// memory of the stack with the requirements req which determines its size.
		Ldlt<T,I> ldl;
		bool do_ldlt;
		bool do_symbolic_fact;
		// persistent allocations

		Eigen::Matrix<T, Eigen::Dynamic, 1> g_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> b_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> l_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> u_scaled;
		veg::Vec<I> kkt_nnz_counts;

		// stored in unique_ptr because we need a stable address
		std::unique_ptr<detail::AugmentedKkt<T, I>> matrix_free_kkt; // view on active part of the KKT which includes the regularizations 
		std::unique_ptr<Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>>
				matrix_free_solver; //eigen based method which takes in entry vector, and performs matrix vector products

		auto stack_mut() -> veg::dynstack::DynStackMut {
			return {
					veg::from_slice_mut,
					storage.as_mut(),
			}; 
		}// exploits all available memory in storage 

	} internal;

	isize lnnz;
	/*!
	 * Constructor using the symbolic factorization.
	 * @param results solver's results.
	 * @param data solver's model.
	 * @param settings solver's settings.
	 * @param precond_req storage requirements for the solver's preconditioner.
	 * @param H symbolic structure of the quadratic cost input defining the QP model.
	 * @param A symbolic structure of the equality constraint matrix input defining the QP model.
	 * @param C symbolic structure of the inequality constraint matrix input defining the QP model.
	 */
	void setup_symbolic_factorizaton(
			Results<T>& results,
			Model<T, I>& data,
			Settings<T>& settings,
			veg::dynstack::StackReq precond_req,
			linearsolver::sparse::SymbolicMatRef<I> H,linearsolver::sparse::SymbolicMatRef<I> AT,linearsolver::sparse::SymbolicMatRef<I> CT
			){
		auto& ldl = internal.ldl;
		
		auto& storage = internal.storage ;
		auto& do_ldlt = internal.do_ldlt;
		// persistent allocations

		data.dim = H.nrows();
		data.n_eq = AT.ncols();
		data.n_in = CT.ncols();
		data.H_nnz = H.nnz();
		data.A_nnz = AT.nnz();
		data.C_nnz = CT.nnz();

		using namespace veg::dynstack;
		using namespace linearsolver::sparse::util;

		using SR = StackReq;
		veg::Tag<I> itag; // ?
		veg::Tag<T> xtag; // ?

		isize n = H.nrows();
		isize n_eq = AT.ncols();
		isize n_in = CT.ncols();
		isize n_tot = n + n_eq + n_in;

		isize nnz_tot = H.nnz() + AT.nnz() + CT.nnz();

		// form the full kkt matrix
		// assuming H, AT, CT are sorted
		// and H is upper triangular
		{
			data.kkt_col_ptrs.resize_for_overwrite(n_tot + 1); // 
			data.kkt_row_indices.resize_for_overwrite(nnz_tot);
			data.kkt_values.resize_for_overwrite(nnz_tot);

			I* kktp = data.kkt_col_ptrs.ptr_mut();
			I* kkti = data.kkt_row_indices.ptr_mut();

			kktp[0] = 0;
			usize col = 0;
			usize pos = 0;

			auto insert_submatrix = [&](linearsolver::sparse::SymbolicMatRef<I> m,
										bool assert_sym_hi) -> void {
				I const* mi = m.row_indices();
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

						++pos;
					}
				}
			};

			insert_submatrix(H, true);
			insert_submatrix(AT, false);
			insert_submatrix(CT, false);
		}

		data.kkt_col_ptrs_unscaled = data.kkt_col_ptrs;
		data.kkt_row_indices_unscaled = data.kkt_row_indices;


		storage.resize_for_overwrite( //
				(StackReq::with_len(itag, n_tot) &
				linearsolver::sparse::factorize_symbolic_req( //
							itag,                                     //
							n_tot,                                    //
							nnz_tot,                                  //
							linearsolver::sparse::Ordering::amd))     //
						.alloc_req()                               //
		); 

		ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
		ldl.perm_inv.resize_for_overwrite(n_tot);

		DynStackMut stack = stack_mut();

		bool overflow = false;
		{
			ldl.etree.resize_for_overwrite(n_tot);
			auto etree_ptr = ldl.etree.ptr_mut();

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
					ldl.col_ptrs.ptr_mut() + 1,// reimplements col counts to get the matrix free version as well
					etree_ptr,
					ldl.perm_inv.ptr_mut(),
					static_cast<I const*>(nullptr),
					kkt_sym,
					stack);
			
			auto pcol_ptrs = ldl.col_ptrs.ptr_mut();
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

		lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));

		// if ldlt is too sparse
		// do_ldlt = !overflow && lnnz < (10000000);
		do_ldlt = !overflow && lnnz < 10000000;
		
		internal.do_symbolic_fact = false;
	}
	/*!
	 * Constructor.
	 * @param qp view on the qp problem.
	 * @param results solver's results.
	 * @param data solver's model.
	 * @param settings solver's settings.
	 * @param execute_or_not boolean option for execturing or not the preconditioner for scaling the problem (and reduce its ill conditioning).
	 * @param precond preconditioner chosen for the solver.
	 * @param precond_req storage requirements for the solver's preconditioner.
	 */
	template <typename P>
	void setup_impl(
			QpView<T, I> qp,
			Results<T>& results,
			Model<T, I>& data,
			Settings<T>& settings,
			bool execute_or_not,
			P& precond,
			veg::dynstack::StackReq precond_req) {
		
		auto& ldl = internal.ldl;
		
		auto& storage = internal.storage ;
		auto& do_ldlt = internal.do_ldlt;
		// persistent allocations

		auto& g_scaled = internal.g_scaled;
		auto& b_scaled = internal.b_scaled;
		auto& l_scaled = internal.l_scaled;
		auto& u_scaled = internal.u_scaled;
		auto& kkt_nnz_counts = internal.kkt_nnz_counts;

		// stored in unique_ptr because we need a stable address
		auto& matrix_free_solver = internal.matrix_free_solver;
		auto& matrix_free_kkt = internal.matrix_free_kkt;

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

		if (internal.do_symbolic_fact){

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

		data.kkt_col_ptrs_unscaled = data.kkt_col_ptrs;
		data.kkt_row_indices_unscaled = data.kkt_row_indices;
		data.kkt_values_unscaled = data.kkt_values;


		storage.resize_for_overwrite( //
				(StackReq::with_len(itag, n_tot) &
				linearsolver::sparse::factorize_symbolic_req( //
							itag,                                     //
							n_tot,                                    //
							nnz_tot,                                  //
							linearsolver::sparse::Ordering::amd))     //
						.alloc_req()                               //
		);

		ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
		ldl.perm_inv.resize_for_overwrite(n_tot);

		DynStackMut stack = stack_mut();

		bool overflow = false;
		{
			ldl.etree.resize_for_overwrite(n_tot);
			auto etree_ptr = ldl.etree.ptr_mut();

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
					ldl.col_ptrs.ptr_mut() + 1,
					etree_ptr,
					ldl.perm_inv.ptr_mut(),
					static_cast<I const*>(nullptr),
					kkt_sym,
					stack);

			auto pcol_ptrs = ldl.col_ptrs.ptr_mut(); 
			pcol_ptrs[0] = I(0);//pcol_ptrs +1: pointor towards the nbr of non zero elts per column of the ldlt 
			// we need to compute its cumulative sum below to determine if there could be an overflow

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

		auto lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));

		// if ldlt is too sparse
		// do_ldlt = !overflow && lnnz < (10000000);
		do_ldlt = !overflow && lnnz < 10000000;
	}else{
			T* kktx = data.kkt_values.ptr_mut();
			usize pos = 0;	
			auto insert_submatrix = [&](linearsolver::sparse::MatRef<T, I> m,
										bool assert_sym_hi) -> void {

				T const* mx = m.values();
				isize ncols = m.ncols();

				for (usize j = 0; j < usize(ncols); ++j) {
					usize col_start = m.col_start(j);
					usize col_end = m.col_end(j);
					for (usize p = col_start; p < col_end; ++p) {

						kktx[pos] = mx[p];

						++pos;
					}
				}
			};

			insert_submatrix(qp.H, true);
			insert_submatrix(qp.AT, false);
			insert_submatrix(qp.CT, false);
			data.kkt_values_unscaled = data.kkt_values;
	}
#define PROX_QP_ALL_OF(...)                                                    \
::veg::dynstack::StackReq::and_(::veg::init_list(__VA_ARGS__))
#define PROX_QP_ANY_OF(...)                                                    \
::veg::dynstack::StackReq::or_(::veg::init_list(__VA_ARGS__))
		//  ? --> if
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
									SR::with_len(itag, 0), // compute necessary space for storing n elts of type I (n = 0 here)
									SR::with_len(xtag, 0), // compute necessary space for storing n elts of type T (n = 0 here)
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
		// define memory needed for primal_dual_newton_semi_smooth
		//PROX_QP_ALL_OF --> need to store all argument inside
		//PROX_QP_ANY_OF --> au moins un de  ceux en entrée
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

		storage.resize_for_overwrite(req.alloc_req()); // defines the maximal storage size 
		// storage.resize(n): if it is done twice in a row, the second times it does nothing, as the same resize has been asked

		// preconditioner
		auto kkt = data.kkt_mut();
		auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n); //  top_rows_mut_unchecked: take a view of sparse matrix for n first lines ; the function assumes all others lines are zeros;
		/*
			H AT CT
			A 
			C

			here we store the upper triangular part below

			tirSup(H) AT CT
			0 0 0 
			0 0 0 

			veg::unsafe:  precises that the function has undefined behavior if upper condition is not respected.
		*/

		linearsolver::sparse::MatMut<T, I> H_scaled =
				detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);

		linearsolver::sparse::MatMut<T, I> AT_scaled =
				detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, data.A_nnz);

		linearsolver::sparse::MatMut<T, I> CT_scaled =
				detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, data.C_nnz);

		g_scaled = data.g;
		b_scaled = data.b;
		l_scaled = data.l;
		u_scaled = data.u;

		QpViewMut<T, I> qp_scaled = {
				H_scaled,
				{linearsolver::sparse::from_eigen, g_scaled},
				AT_scaled,
				{linearsolver::sparse::from_eigen, b_scaled},
				CT_scaled,
				{linearsolver::sparse::from_eigen, l_scaled},
				{linearsolver::sparse::from_eigen, u_scaled},
		};
		
		DynStackMut stack = stack_mut();
		precond.scale_qp_in_place(qp_scaled, execute_or_not, settings, stack);

		kkt_nnz_counts.resize_for_overwrite(n_tot);

		linearsolver::sparse::MatMut<T, I> kkt_active = {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				data.H_nnz + data.A_nnz,// these variables are not used for the matrix vector product in augmented KKT with Min res algorithm (to be exact, it should depend of the initial guess)
				kkt.col_ptrs_mut(),
				kkt_nnz_counts.ptr_mut(),
				kkt.row_indices_mut(),
				kkt.values_mut(),
		};

		using MatrixFreeSolver = Eigen::MINRES<
				detail::AugmentedKkt<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>;
		matrix_free_solver = std::unique_ptr<MatrixFreeSolver>{
				new MatrixFreeSolver,
		};
		matrix_free_kkt = std::unique_ptr<detail::AugmentedKkt<T, I>>{
				new detail::AugmentedKkt<T, I>{
						{
								kkt_active.as_const(),
								{},
								n,
								n_eq,
								n_in,
								{},
								{},
								{},
						},
				}};

		auto zx = linearsolver::sparse::util::zero_extend;// ?
		auto max_lnnz = isize(zx(ldl.col_ptrs[n_tot]));
		isize ldlt_ntot = do_ldlt ? n_tot : 0;
		isize ldlt_lnnz = do_ldlt ? max_lnnz : 0;

		ldl.nnz_counts.resize_for_overwrite(ldlt_ntot);
		ldl.row_indices.resize_for_overwrite(ldlt_lnnz);
		ldl.values.resize_for_overwrite(ldlt_lnnz);
		
		ldl.perm.resize_for_overwrite(ldlt_ntot);
		if (do_ldlt) {
			// compute perm from perm_inv
			for (isize i = 0; i < n_tot; ++i) {
				ldl.perm[isize(zx(ldl.perm_inv[i]))] = I(i);
			}
		}
	}
	Timer<T> timer;
	Workspace() = default;

	auto ldl_col_ptrs() const -> I const* {
		return internal.ldl.col_ptrs.ptr();
	}
	auto ldl_col_ptrs_mut() -> I* {
		return internal.ldl.col_ptrs.ptr_mut();
	}
	auto stack_mut() -> veg::dynstack::DynStackMut {
		return internal.stack_mut();
	}
};

} //namespace sparse
} //namespace qp
} //namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WORKSPACE_HPP */
