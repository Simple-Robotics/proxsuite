//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_QP_SPARSE_SOLVER_OSQP_HPP
#define PROXSUITE_QP_SPARSE_SOLVER_OSQP_HPP

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

template <typename T>
using VecMapMut = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1>,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;
template <typename T>
using VecMap = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1> const,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;

template <typename T, typename I, typename P>
auto osqp_unscaled_primal_dual_residual(
		VecMapMut<T> primal_residual_eq_scaled,
		VecMapMut<T> primal_residual_in_scaled_lo,
		VecMapMut<T> primal_residual_in_scaled_up,
		VecMapMut<T> dual_residual_scaled,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T dual_feasibility_rhs_0,
		T dual_feasibility_rhs_1,
		T dual_feasibility_rhs_3,
		P& precond,
		Model<T, I> const& data,
		QpView<T, I> qp_scaled,
		VecMap<T> x_e,
		VecMap<T> y_e,
		veg::dynstack::DynStackMut stack) -> veg::Tuple<T, T> {


    // dual

	isize n = x_e.rows();
    isize n_eq = primal_residual_eq_scaled.rows();
    isize n_in = primal_residual_in_scaled_lo.rows();
	primal_residual_eq_scaled.setZero();
	primal_residual_in_scaled_lo.setZero();
	primal_residual_in_scaled_up.setZero();

	LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
	dual_residual_scaled = qp_scaled.g.to_eigen();

	{
		tmp.setZero();
		
		detail::noalias_symhiv_add(tmp, qp_scaled.H.to_eigen(), x_e);
		dual_residual_scaled += tmp; // contains now scaled(g+Hx)

		precond.unscale_dual_residual_in_place({qp::from_eigen, tmp});
		dual_feasibility_rhs_0 = infty_norm(tmp); // ||unscaled(Hx)||
	}

	{
		auto ATy = tmp;
		ATy.setZero();
		
		detail::noalias_gevmmv_add(
				primal_residual_eq_scaled, ATy, qp_scaled.AT.to_eigen(), x_e, y_e.head(n_eq));
		dual_residual_scaled += ATy; // contains now scaled(g+Hx+ATy)

		precond.unscale_dual_residual_in_place({qp::from_eigen, ATy});
		dual_feasibility_rhs_1 = infty_norm(ATy); // ||unscaled(ATy)||
	}

	{
		auto CTz = tmp;
		
		CTz.setZero();
		detail::noalias_gevmmv_add(
				primal_residual_in_scaled_up, CTz, qp_scaled.CT.to_eigen(), x_e, y_e.tail(n_in));
		dual_residual_scaled += CTz; // contains now scaled(g+Hx+ATy+CTz)

		precond.unscale_dual_residual_in_place({qp::from_eigen, CTz});
		dual_feasibility_rhs_3 = infty_norm(CTz); // ||unscaled(CTz)||
	}


	precond.unscale_dual_residual_in_place(
			{qp::from_eigen, dual_residual_scaled}); // contains now unscaled(Hx+g+ATy+CTz)
	T dual_feasibility_lhs = infty_norm(dual_residual_scaled); // ||unscaled(Hx+g+ATy+CTz)||
	precond.scale_dual_residual_in_place({qp::from_eigen, dual_residual_scaled});// ||scaled(Hx+g+ATy+CTz)||


    // primal 
	auto b = data.b;
	auto l = data.l;
	auto u = data.u;

	precond.unscale_primal_residual_in_place_eq(
			{qp::from_eigen, primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm(primal_residual_eq_scaled); // ||unscaled(Ax)||

	precond.unscale_primal_residual_in_place_in(
			{qp::from_eigen, primal_residual_in_scaled_up});
	std::cout << "primal_residual_in_scaled_up before norming " << primal_residual_in_scaled_up << std::endl;
	primal_feasibility_in_rhs_0 = infty_norm(primal_residual_in_scaled_up); // ||unscaled(Cx)||

	primal_residual_in_scaled_lo =
			detail::positive_part(primal_residual_in_scaled_up - u) +
			detail::negative_part(primal_residual_in_scaled_up - l);
	primal_residual_eq_scaled -= b;

	T primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_scaled);
	T primal_feasibility_in_lhs = infty_norm(primal_residual_in_scaled_lo);
	T primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	// scaled Ax - b
	precond.scale_primal_residual_in_place_eq(
			{qp::from_eigen, primal_residual_eq_scaled});
    // scaled Cx
	precond.scale_primal_residual_in_place_in(
			{qp::from_eigen, primal_residual_in_scaled_up});


	return veg::tuplify(primal_feasibility_lhs, dual_feasibility_lhs);
}

template <typename T>
struct OSQPInfo {
    ///// final proximal regularization parameters
    T mu_eq;
    T mu_eq_inv;
    T mu_in;
    T mu_in_inv;
    T rho;
	T kappa;				

    ///// iteration count
    isize iter;
    isize mu_updates;

    //// timings
    T solve_time;
    T objValue;
};

template <typename T>
struct OSQPResults {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// SOLUTION STORAGE

    Vec x;
    Vec y;
    Vec z;

	OSQPInfo<T> info;

	OSQPResults( isize dim=0, isize n_eq=0, isize n_in=0)
			: //
                x(dim),
                y(n_in+n_eq),
                z(n_eq+n_in)
                {
        
                x.setZero();
                y.setZero();
                z.setZero();

                info.rho = 1e-8;
				info.kappa = 1.E1;
	            info.mu_eq = 1e-3;
	            info.mu_eq_inv = 1e3 ;
	            info.mu_in = 1e-1;
	            info.mu_in_inv = 1e1;

                info.iter = 0;
                info.mu_updates = 0;
                info.solve_time = 0.;
                info.objValue =0.;
                
                }
    
    void reset_results(){
        x.setZero();
        y.setZero();
        z.setZero();

		info.rho = 1e-8;
		info.kappa = 1.E1;
		info.mu_eq = 1e-3;
		info.mu_eq = 1e-3 ;
		info.mu_in = 1e-1;
		info.mu_in_inv = 1e-1;

		info.iter = 0;
		info.mu_updates = 0;
		info.solve_time = 0.;
		info.objValue =0.;
                
    }
};

template <typename T,typename I>
void ldl_solve(VectorViewMut<T> sol, VectorView<T> rhs,isize n_tot,
	linearsolver::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,veg::dynstack::DynStackMut stack,
			T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv){
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

template <typename T,typename I>
void ldl_iter_solve_noalias(VectorViewMut<T> sol,
									VectorView<T> rhs,
									VectorView<T> init_guess,
									OSQPResults<T> results,
									Model<T,I> data,
									isize n_tot,
									linearsolver::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,veg::dynstack::DynStackMut stack,T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv,
			Settings<T> const& settings){
		auto rhs_e = rhs.to_eigen();
		auto sol_e = sol.to_eigen();

		if (init_guess.dim == sol.dim) {
			sol_e = init_guess.to_eigen();
		} else {
			sol_e.setZero();
		}

		LDLT_TEMP_VEC_UNINIT(T, err, n_tot, stack);
		linearsolver::sparse::MatMut<T, I> kkt = data.kkt_mut(); // scaled in the setup

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
				detail::noalias_symhiv_add(err, kkt.to_eigen(), sol_e); // replaced kkt_active by kkt
				err_x += results.info.rho * sol_x;
				err_y += mu_eq_neg * sol_y;
                err_z += mu_in_neg * sol_z; // removed active set condition
			}

			T err_norm = infty_norm(err);
			if (err_norm > prev_err_norm / T(2)) {
				break;
			}
			prev_err_norm = err_norm;

			ldl_solve({qp::from_eigen, err}, {qp::from_eigen, err},n_tot,ldl,iterative_solver,do_ldlt,stack,ldl_values,perm,ldl_col_ptrs,perm_inv);

			sol_e -= err;
		}
};

template<typename T, typename I>
void ldl_solve_in_place(VectorViewMut<T> rhs,
						VectorView<T> init_guess,
						OSQPResults<T> results,
						Model<T,I> data,
						isize n_tot,
						linearsolver::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,veg::dynstack::DynStackMut stack,T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv,
			Settings<T> const& settings) {
	LDLT_TEMP_VEC_UNINIT(T, tmp, n_tot, stack);
	ldl_iter_solve_noalias({qp::from_eigen, tmp}, rhs.as_const(), init_guess,results, data, n_tot, ldl, iterative_solver,do_ldlt,stack,ldl_values,perm,ldl_col_ptrs,perm_inv,settings);
	rhs.to_eigen() = tmp;
};

template<typename T>
using DMat = Eigen::Matrix<T, -1, -1>;

template<typename T, typename I>
DMat<T> inner_reconstructed_matrix(linearsolver::sparse::MatMut<T, I> ldl,bool do_ldlt){
	VEG_ASSERT(do_ldlt);
	auto ldl_dense = ldl.to_eigen().toDense();
	auto l = DMat<T>(ldl_dense.template triangularView<Eigen::UnitLower>());
	auto lt = l.transpose();
	auto d = ldl_dense.diagonal().asDiagonal();
	auto mat = DMat<T>(l * d * lt);
	return mat;
};

template<typename T,typename I>
DMat<T> reconstructed_matrix(linearsolver::sparse::MatMut<T, I> ldl,bool do_ldlt,I const* perm_inv,isize n_tot){
	auto mat = inner_reconstructed_matrix(ldl,do_ldlt);
	auto mat_backup = mat;
	for (isize i = 0; i < n_tot; ++i) {
		for (isize j = 0; j < n_tot; ++j) {
			mat(i, j) = mat_backup(perm_inv[i], perm_inv[j]);
		}
	}
	return mat;
};

template<typename T,typename I>
DMat<T>  reconstruction_error(linearsolver::sparse::MatMut<T, I> ldl,bool do_ldlt,I const* perm_inv,OSQPResults<T> results,Model<T,I> data,isize n_tot){
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	linearsolver::sparse::MatMut<T, I> kkt = data.kkt_mut();
	auto diff = DMat<T>(
			reconstructed_matrix(ldl,do_ldlt,perm_inv,n_tot) -
			DMat<T>(DMat<T>(kkt.to_eigen())
					.template selfadjointView<Eigen::Upper>()));
	diff.diagonal().head(data.dim).array() -= results.info.rho;
	diff.diagonal().segment(data.dim, data.n_eq).array() -= mu_eq_neg;
	for (isize i = 0; i < data.n_in; ++i) {
		diff.diagonal()[data.dim + data.n_eq + i] -=  mu_in_neg;
	}
	return diff;
};

template <typename T, typename I, typename P>
void osqp_solve(
		OSQPResults<T>& results,
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

	linearsolver::sparse::MatMut<T, I> kkt = data.kkt_mut(); // scaled in the setup

	//// le scaling a été enlevé

	auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n);

	linearsolver::sparse::MatMut<T, I> H_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);

	linearsolver::sparse::MatMut<T, I> AT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, data.A_nnz);

	linearsolver::sparse::MatMut<T, I> CT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, data.C_nnz);

	LDLT_TEMP_VEC_UNINIT(T, g_scaled_e, n, stack);
	LDLT_TEMP_VEC_UNINIT(T, b_scaled_e, n_eq, stack);
	LDLT_TEMP_VEC_UNINIT(T, l_scaled_e, n_in, stack);
	LDLT_TEMP_VEC_UNINIT(T, u_scaled_e, n_in, stack);

	g_scaled_e = data.g;
	b_scaled_e = data.b;
	l_scaled_e = data.l;
	u_scaled_e = data.u;

	QpViewMut<T, I> qp_scaled = {
			H_scaled,
			{linearsolver::sparse::from_eigen, g_scaled_e},
			AT_scaled,
			{linearsolver::sparse::from_eigen, b_scaled_e},
			CT_scaled,
			{linearsolver::sparse::from_eigen, l_scaled_e},
			{linearsolver::sparse::from_eigen, u_scaled_e},
	};

	precond.scale_qp_in_place(qp_scaled, stack); // a view on kkt hence it scales in place as well kkt

	////

	T const primal_feasibility_rhs_1_eq = infty_norm(data.b);
	T const primal_feasibility_rhs_1_in_u = infty_norm(data.u);
	T const primal_feasibility_rhs_1_in_l = infty_norm(data.l);
	T const dual_feasibility_rhs_2 = infty_norm(data.g);

	auto ldl_col_ptrs = work.ldl_col_ptrs_mut();
	auto max_lnnz = isize(zx(ldl_col_ptrs[n_tot]));

	veg::Tag<I> itag;
	veg::Tag<T> xtag;

	auto _active_constraints = stack.make_new(veg::Tag<bool>{}, n_in).unwrap();
	//auto _kkt_nnz_counts = stack.make_new_for_overwrite(itag, n_tot).unwrap();

	bool do_ldlt = work._.do_ldlt;

	isize ldlt_ntot = do_ldlt ? n_tot : 0;
	isize ldlt_lnnz = do_ldlt ? max_lnnz : 0;

	auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _etree = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _ldl_nnz_counts = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _ldl_row_indices =
			stack.make_new_for_overwrite(itag, ldlt_lnnz).unwrap();
	auto _ldl_values = stack.make_new_for_overwrite(xtag, ldlt_lnnz).unwrap();

	I const* perm_inv = work._.perm_inv.ptr();
	I* perm = _perm.ptr_mut();

	if (do_ldlt) {
		// compute perm from perm_inv
		for (isize i = 0; i < n_tot; ++i) {
			perm[isize(zx(perm_inv[i]))] = I(i);
		}
	}

	Eigen::MINRES<
			detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>
			iterative_solver;

	I* etree = _etree.ptr_mut();
	I* ldl_nnz_counts = _ldl_nnz_counts.ptr_mut();
	I* ldl_row_indices = _ldl_row_indices.ptr_mut();
	T* ldl_values = _ldl_values.ptr_mut();
	veg::SliceMut<bool> active_constraints = _active_constraints.as_mut();
	for (isize i = 0; i < data.n_in; ++i){
		active_constraints[i] = true;
	}

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

	detail::AugmentedKkt<T, I> aug_kkt{{
			kkt.as_const(), // replaced kkt_active by kkt
			active_constraints.as_const(),
			n,
			n_eq,
			n_in,
			results.info.rho,
			results.info.mu_eq_inv,
			results.info.mu_in_inv
	}};

	auto refactorize = [&]() -> void {
		T mu_eq_neg = -results.info.mu_eq;
		T mu_in_neg = -results.info.mu_in;
		if (do_ldlt) {
			linearsolver::sparse::factorize_symbolic_non_zeros(
					ldl_nnz_counts,
					etree,
					work._.perm_inv.ptr_mut(),
					perm,
					kkt.symbolic(),// replaced kkt_active by kkt
					stack);

			auto _diag = stack.make_new_for_overwrite(xtag, n_tot).unwrap();
			T* diag = _diag.ptr_mut();

			for (isize i = 0; i < n; ++i) {
				diag[i] = results.info.rho;
			}
			for (isize i = 0; i < n_eq; ++i) {
				diag[n + i] = mu_eq_neg;
			}
			for (isize i = 0; i < n_in; ++i) { // change here
				diag[(n + n_eq) + i] = mu_in_neg;
			}

			linearsolver::sparse::factorize_numeric(
					ldl_values,
					ldl_row_indices,
					diag,
					perm,
					ldl_col_ptrs,
					etree,
					perm_inv,
					kkt.as_const(), // replaced kkt_active by kkt
					stack);
			isize ldl_nnz = 0;
			for (isize i = 0; i < n_tot; ++i) {
				ldl_nnz =
						util::checked_non_negative_plus(ldl_nnz, isize(ldl_nnz_counts[i]));
			}
			ldl._set_nnz(ldl_nnz);
		} else {
			iterative_solver.compute(aug_kkt);
		}
	};
	refactorize();
	//std::cout << "reconstruction error " << infty_norm(reconstruction_error()) << std::endl;
	std::cout<< "reconstruction error " << infty_norm(reconstruction_error(ldl,do_ldlt,perm_inv,results,data,n_tot,kkt_active,active_constraints)) << std::endl;

	auto x_e = x.to_eigen();
	auto y_e = y.to_eigen();
	auto z_e = z.to_eigen();

	if (!settings.warm_start) {
		LDLT_TEMP_VEC_UNINIT(T, rhs, n_tot, stack);
		LDLT_TEMP_VEC_UNINIT(T, no_guess, 0, stack);

		rhs.head(n) = -g_scaled_e;
		rhs.segment(n, n_eq) = b_scaled_e;
		rhs.segment(n + n_eq, n_in).setZero();

		ldl_solve_in_place({qp::from_eigen, rhs}, {qp::from_eigen, no_guess},results,data, n_tot, ldl, iterative_solver,do_ldlt,stack,ldl_values,perm,ldl_col_ptrs,perm_inv,settings);
		x_e = rhs.head(n);
		y_e = rhs.tail(n_eq+n_in);
	}

	x_e.setZero();
	y_e.setZero();
	z_e.setZero(); // check if warm start or not
	for (isize iter = 0; iter < settings.max_iter; ++iter) {

		T new_bcl_mu_eq = results.info.mu_eq;
		T new_bcl_mu_in = results.info.mu_in;
		T new_bcl_mu_eq_inv = results.info.mu_eq_inv;
		T new_bcl_mu_in_inv = results.info.mu_in_inv;

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
				osqp_unscaled_primal_dual_residual(
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
						stack));

		if (settings.verbose) {
			std::cout << "-------- outer iteration: " << iter << " primal residual "
								<< primal_feasibility_lhs << " dual residual "
								<< dual_feasibility_lhs << " mu_in " << results.info.mu_in << " mu_eq " << results.info.mu_eq << std::endl;
		}
		if (is_primal_feasible(primal_feasibility_lhs) &&
			is_dual_feasible(dual_feasibility_lhs)) {
			break;
		}

		LDLT_TEMP_VEC_UNINIT(T, x_prev_e, n, stack);
		LDLT_TEMP_VEC_UNINIT(T, y_prev_e, n_eq + n_in, stack);
		LDLT_TEMP_VEC_UNINIT(T, z_prev_e, n_eq + n_in, stack);
		LDLT_TEMP_VEC(T, dw_prev, n_tot, stack);

		x_prev_e = x_e;
		y_prev_e = y_e;
		z_prev_e = z_e;

		// USING OSQP simplification
		primal_residual_in_scaled_up.noalias() -= z_prev_e.tail(data.n_in); // contains now scaled(Cx - ze) 

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		
		LDLT_TEMP_VEC_UNINIT(T, dw, n_tot, stack);
		T alpha = 1.6;

		// Newton step
	
		/// according to osqp simplification
		// z_eq == b
		//
		// [ H + rho I    AT          CT  ]       [ H×x + g + AT×y_eq + CT×y_in ]
		// [ A         -µ_eq I      0     ]       [ A×x - z_eq                  ]
		// [ C            0        -µ_in I] dw = -[ C×x - z_in                  ]

		auto rhs = dw;

		// using osqp simplification
		rhs.head(n).noalias() = -dual_residual_scaled;
		rhs.segment(n, n_eq).noalias() = -primal_residual_eq_scaled ; 
		rhs.tail(n_in).noalias() = -primal_residual_in_scaled_up;

		ldl_solve_in_place(
				{qp::from_eigen, rhs}, {qp::from_eigen, dw_prev},results,data,n_tot,ldl,iterative_solver,do_ldlt,stack,ldl_values,perm,ldl_col_ptrs,perm_inv,settings);
	
		// ITERATES UPDATES according to OSQP simplification
		// tmp = alpha * µ_in dw + zk + yk µ_in: we treat only n_in equalities as the projection of tmp on equalities is only a point (b)
		LDLT_TEMP_VEC_UNINIT(T, tmp, data.n_in, stack);
		tmp.noalias() = (alpha * results.info.mu_in) * dw.tail(n_in) //
								+ z_prev_e.tail(n_in)
								+ y_prev_e.tail(n_in) * results.info.mu_in;
		z_e.tail(n_in) = tmp + //
					detail::positive_part(l_scaled_e - tmp) -
					detail::positive_part(tmp - u_scaled_e);
		// y{k+1} = yk + µ_in_inv ( alpha (zk + dw/µ) + (1-alpha) zk - z{k+1} )
		//        = yk + µ_in_inv ( zk - z{k+1} + alpha * µ_in dw )
		//        = yk + µ_in_inv (zk - z{k+1}) + alpha dw
		//
		// eq constraints: z_k == z_{k+1} == b
		y_e.head(n_eq) += alpha * dw.segment(n,n_eq);
		y_e.tail(n_in) = results.info.mu_in_inv * (tmp - z_e.tail(n_in));
		x_e += alpha * dw.head(n);

	
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		/*
		VEG_BIND(
				auto,
				(primal_feasibility_lhs_new, dual_feasibility_lhs_new),
				osqp_unscaled_primal_dual_residual(
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
						stack));
		if (is_primal_feasible(primal_feasibility_lhs_new) &&
			is_dual_feasible(dual_feasibility_lhs_new)) {
			break;
		}

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		// TODO mu update
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		if (results.info.mu_in != new_bcl_mu_in ||
		    results.info.mu_eq != new_bcl_mu_eq) {
			{ ++results.info.mu_updates; }
			refactorize();
		}

		results.info.mu_eq = new_bcl_mu_eq;
		results.info.mu_in = new_bcl_mu_in;
		results.info.mu_eq_inv = new_bcl_mu_eq_inv;
		results.info.mu_in_inv = new_bcl_mu_in_inv;
		*/
	}

	LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
	tmp.setZero();
	detail::noalias_symhiv_add(tmp, qp_scaled.H.to_eigen(), x_e);
	precond.unscale_dual_residual_in_place({qp::from_eigen, tmp});
	
	precond.unscale_primal_in_place({qp::from_eigen, x_e});
	precond.unscale_dual_in_place({qp::from_eigen, y_e});

	tmp *= 0.5;
	tmp += data.g;
	results.info.objValue = (tmp).dot(x_e);

}

template <typename T,typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

template <typename T>
void osqp_update_proximal_parameters(
		OSQPResults<T>& results,
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

template <typename T>
void osqp_warm_starting(
		tl::optional<VecRef<T>> x_wm,
		tl::optional<VecRef<T>> y_wm,
		tl::optional<VecRef<T>> z_wm,
		OSQPResults<T>& results,
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
void osqp_setup(
		QpView<T, I> qp,
		OSQPResults<T>& results,
		Model<T, I>& data,
		Workspace<T, I>& work,
		P& precond) {
	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();

	if (results.x.rows() != n) {
		results.x.resize(n);
		results.x.setZero();
	}
	if (results.y.rows() != n_eq+n_in) {
		results.y.resize(n_eq+n_in);
		results.y.setZero();
	}
	if (results.z.rows() != n_in+n_eq) {
		results.z.resize(n_in+n_in);
		results.z.setZero();
	}

	work._.setup_impl(
			qp, data, P::scale_qp_in_place_req(veg::Tag<T>{}, n, n_eq, n_in));

}
///// QP object
template <typename T,typename I>
struct OSQP {
public:
	OSQPResults<T> results;
	Settings<T> settings;
	Model<T,I> data;
	Workspace<T,I> work;
    preconditioner::RuizEquilibration<T, I> ruiz;

	OSQP(isize _dim, isize _n_eq, isize _n_in)
			: results(_dim, _n_eq, _n_in),
				settings(),
				data(),
				work(),ruiz(_dim,_n_eq + _n_in,1e-3,10,preconditioner::Symmetry::UPPER) {}

	void setup_sparse_matrices(
			const tl::optional<SparseMat<T,I>> H,
			tl::optional<VecRef<T>> g,
			const tl::optional<SparseMat<T,I>> A,
			tl::optional<VecRef<T>> b,
			const tl::optional<SparseMat<T,I>> C,
			tl::optional<VecRef<T>> u,
			tl::optional<VecRef<T>> l) {
                
        SparseMat<T, I> H_triu = H.value().template triangularView<Eigen::Upper>();
        // only initial setup available (if an update of only one)

        SparseMat<T, I> AT = A.value().transpose();
        SparseMat<T, I> CT = C.value().transpose();
        sparse::QpView<T, I> qp = {
            {linearsolver::sparse::from_eigen, H_triu},
            {linearsolver::sparse::from_eigen, g.value()},
            {linearsolver::sparse::from_eigen, AT},
            {linearsolver::sparse::from_eigen, b.value()},
            {linearsolver::sparse::from_eigen, CT},
            {linearsolver::sparse::from_eigen, l.value()},
            {linearsolver::sparse::from_eigen, u.value()}};
        
        osqp_setup(
                qp,
                results,
                data,
                work,
                ruiz);
	};

	void solve() {

		auto start = std::chrono::high_resolution_clock::now();
		sparse::osqp_solve( //
				results,
				data,
				settings,
				work,
                ruiz);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration =
				std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		results.info.solve_time = T(duration.count());
		//results.info.run_time = results.info.solve_time + results.info.setup_time;

		if (settings.verbose) {
			std::cout << "------ SOLVER STATISTICS--------" << std::endl;
			// TODO
		}
	};

	void update_prox_parameter(
			tl::optional<T> rho, tl::optional<T> mu_eq, tl::optional<T> mu_in) {
		osqp_update_proximal_parameters(results, rho, mu_eq, mu_in);
	};
	void warm_start(
			tl::optional<VecRef<T>> x,
			tl::optional<VecRef<T>> y,
			tl::optional<VecRef<T>> z) {
		osqp_warm_starting(x, y, z, results, settings);
	};
	void cleanup() {
		results.reset_results();
	}
};



} // namespace sparse
} // namespace qp
} // namespace proxsuite


#endif /* end of include guard PROXSUITE_QP_SPARSE_SOLVER_OSQP_HPP */
