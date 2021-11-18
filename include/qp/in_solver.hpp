#ifndef INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/line_search.hpp"
#include "cnpy.hpp"
#include "qp/precond/identity.hpp"
#include <cmath>
#include <type_traits>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {
using namespace cnpy;
struct QpSolveStats {
	isize n_ext;
	isize n_mu_updates;
	isize n_tot;
};

#define LDLT_DEDUCE_RET(...)                                                   \
	noexcept(noexcept(__VA_ARGS__))                                              \
			->typename std::remove_const<decltype(__VA_ARGS__)>::type {              \
		return __VA_ARGS__;                                                        \
	}                                                                            \
	static_assert(true, ".")

template <typename T>
auto positive_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		LDLT_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

template <typename T>
void refactorize(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		MatrixViewMut<T, colmajor> kkt,
		isize n_c,
		T mu_in,
		T rho_old,
		T rho_new,
		ldlt::Ldlt<T>& ldl) {
	auto current_bijection_map = current_bijection_map_.to_eigen();

	isize const dim = qp_scaled.H.rows;
	isize const n_eq = qp_scaled.A.rows;
	isize const n_in = qp_scaled.C.rows;

	auto Htot = kkt.to_eigen();
	for (isize i = 0; i < dim; ++i) {
		Htot(i, i) += rho_new - rho_old;
	}

	ldl.factorize(Htot);

	if (n_c == 0) {
		return;
	}

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_new_col, Init, Vec(n_eq + dim + n_c), LDLT_CACHELINE_BYTES, T));

	for (isize j = 0; j < n_c; ++j) {
		for (isize i = 0; i < n_in; ++i) {
			if (j == current_bijection_map(i)) {
				auto new_col = _new_col.segment(0, dim + n_eq + j + 1).to_eigen();
				new_col.topRows(dim) = qp_scaled.C.to_eigen().row(i);
				new_col(dim + n_eq + j) = -T(1) / mu_in;
				ldl.insert_at(n_eq + dim + j, new_col);
				new_col(dim + n_eq + j) = T(0);
			}
		}
	}
}

template <typename T>
void iterative_residual_osqp(
		qp::QpViewBox<T> qp_scaled,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> err_,
		T mu_eq,
		T mu_in,
		T rho) {
	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto err = err_.to_eigen();

	err = (-rhs).eval();
	err.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol.topRows(dim) + rho * sol.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol.middleRows(dim, n_eq) +
			(qp_scaled.C).to_eigen().transpose() * sol.tail(n_in);
	err.middleRows(dim, n_eq) += (qp_scaled.A).to_eigen() * sol.topRows(dim) -
	                             sol.middleRows(dim, n_eq) / mu_eq;
	err.tail(n_in) +=
			(qp_scaled.C).to_eigen() * sol.topRows(dim) - sol.tail(n_in) / mu_in;
}

template <typename T>
void iterative_solve_with_permut_fact_osqp( //
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> res_,
		ldlt::Ldlt<T>& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		isize dim,
		isize n_eq,
		isize n_in,
		T mu_eq,
		T mu_in,
		T rho) {

	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto res = res_.to_eigen();

	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);

	qp::detail::iterative_residual_osqp<T>(
			qp_scaled, dim, n_eq, n_in, rhs_, sol_, res_, mu_eq, mu_in, rho);
	std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;

	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;

		res.setZero();
		qp::detail::iterative_residual_osqp<T>(
				qp_scaled, dim, n_eq, n_in, rhs_, sol_, res_, mu_eq, mu_in, rho);
		std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;
	}
}

template <typename T>
void iterative_residual(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorViewMut<T> rhs,
		VectorViewMut<T> err,
		VectorView<T> sol,
		T mu_eq,
		T mu_in,
		T rho) {
	auto current_bijection_map = current_bijection_map_.to_eigen();
	auto err_ = err.to_eigen();
	auto sol_ = sol.to_eigen();

	err_ = -rhs.to_eigen();
	err_.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol_.topRows(dim) + rho * sol_.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol_.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			err_.topRows(dim) += sol_(dim + n_eq + j) * qp_scaled.C.to_eigen().row(i);
			err_(dim + n_eq + j) +=
					(qp_scaled.C.to_eigen().row(i)).dot(sol_.topRows(dim)) -
					sol_(dim + n_eq + j) / mu_in;
		}
	}

	err_.middleRows(dim, n_eq) += (qp_scaled.A).to_eigen() * sol_.topRows(dim) -
	                              sol_.middleRows(dim, n_eq) / mu_eq;
}

template <typename T>
void iterative_residual_QPALM(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> err_,
		VectorView<T> mu_,
		T rho) {
	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto err = err_.to_eigen();
	auto mu = mu_.to_eigen();
	auto current_bijection_map = current_bijection_map_.to_eigen();

	err = (-rhs).eval();
	err.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol.topRows(dim) + rho * sol.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			err.topRows(dim) += sol(dim + n_eq + j) * qp_scaled.C.to_eigen().row(i);
			err(dim + n_eq + j) +=
					(qp_scaled.C.to_eigen().row(i)).dot(sol.topRows(dim)) -
					sol(dim + n_eq + j) / mu(n_eq + i);
		}
	}

	err.middleRows(dim, n_eq).array() +=
			((qp_scaled.A).to_eigen() * sol.topRows(dim)).array() -
			sol.middleRows(dim, n_eq).array() / mu.topRows(n_eq).array();
}

template <typename T>
void iterative_solve_with_permut_fact_new( //
		VectorViewMut<T> rhs,
		VectorViewMut<T> sol,
		VectorViewMut<T> res,
		ldlt::Ldlt<T> const& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		T mu_eq,
		T mu_in,
		T rho) {

	auto rhs_ = rhs.to_eigen();
	auto sol_ = sol.to_eigen();
	auto res_ = res.to_eigen();

	i32 it = 0;
	sol_ = rhs_;
	ldl.solve_in_place(sol_);

	auto compute_iterative_residual = [&] {
		qp::detail::iterative_residual<T>(
				qp_scaled,
				current_bijection_map_,
				dim,
				n_eq,
				n_c,
				n_in,
				{from_eigen, rhs_},
				{from_eigen, res_},
				{from_eigen, sol_},
				mu_eq,
				mu_in,
				rho);
	};

	compute_iterative_residual();
	++it;

	std::cout << "infty_norm(res) " << qp::infty_norm(res_) << std::endl;

	while (infty_norm(res_) >= eps) {
		if (it >= max_it) {
			break;
		}
		++it;
		res_ = -res_;
		ldl.solve_in_place(res_);
		sol_ += res_;

		res_.setZero();
		compute_iterative_residual();
		std::cout << "infty_norm(res) " << qp::infty_norm(res_) << std::endl;
	}
}

template <typename T>
void iterative_solve_with_permut_fact_QPALM( //
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> res_,
		ldlt::Ldlt<T>& ldl,
		T eps,
		isize max_it,
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<isize> current_bijection_map_,
		isize dim,
		isize n_eq,
		isize n_c,
		isize n_in,
		VectorView<T> mu_,
		T rho) {

	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto res = res_.to_eigen();
	auto mu = mu_.to_eigen();

	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);

	qp::detail::iterative_residual_QPALM<T>(
			qp_scaled,
			current_bijection_map_,
			dim,
			n_eq,
			n_c,
			n_in,
			{from_eigen, rhs},
			{from_eigen, sol},
			{from_eigen, res},
			{from_eigen, mu},
			rho);
	std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;

	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;

		res.setZero();
		qp::detail::iterative_residual_QPALM<T>(
				qp_scaled,
				current_bijection_map_,
				dim,
				n_eq,
				n_c,
				n_in,
				{from_eigen, rhs},
				{from_eigen, sol},
				{from_eigen, res},
				{from_eigen, mu},
				rho);
		std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;
	}
}

template <typename T>
void mu_update(
		T mu_eq_old,
		T mu_eq_new,
		T mu_in_old,
		T mu_in_new,
		isize dim,
		isize n_eq,
		isize n_c,
		ldlt::Ldlt<T>& ldl) {
	T diff = T(0);
	LDLT_MULTI_WORKSPACE_MEMORY(
			(e_k_, Init, Vec(dim + n_eq + n_c), LDLT_CACHELINE_BYTES, T));
	auto e_k = e_k_.to_eigen().eval();

	if (n_eq > 0) {
		diff = T(1) / mu_eq_old - T(1) / mu_eq_new;

		for (isize i = 0; i < n_eq; i++) {
			e_k(dim + i) = T(1);
			ldl.rank_one_update(e_k, diff);
			e_k(dim + i) = T(0);
		}
	}
	if (n_c > 0) {
		diff = T(1) / mu_in_old - T(1) / mu_in_new;
		for (isize i = 0; i < n_c; i++) {
			e_k(dim + n_eq + i) = T(1);
			ldl.rank_one_update(e_k, diff);
			e_k(dim + n_eq + i) = T(0);
		}
	}
}

template <typename T>
void BCL_update_fact(
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> y,
		VectorViewMut<T> z,

		isize dim,
		isize n_eq,
		isize n_c,
		ldlt::Ldlt<T>& ldl,
		T beta) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		std::cout << "good step" << std::endl;
		bcl_eta_ext = bcl_eta_ext / pow(bcl_mu_in, beta);
		bcl_eta_in = max2(bcl_eta_in / bcl_mu_in, eps_abs);
	} else {
		std::cout << "bad step" << std::endl;
		y.to_eigen() = ye.to_eigen();
		z.to_eigen() = ze.to_eigen();
		T new_bcl_mu_in = min2(bcl_mu_in * T(10), T(1e8));
		T new_bcl_mu_eq = min2(bcl_mu_eq * T(10), T(1e10));
		if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
			{ ++n_mu_updates; }
		}
		qp::detail::mu_update(
				bcl_mu_eq,
				new_bcl_mu_eq,
				bcl_mu_in,
				new_bcl_mu_in,
				dim,
				n_eq,
				n_c,
				ldl);
		bcl_mu_eq = new_bcl_mu_eq;
		bcl_mu_in = new_bcl_mu_in;
		bcl_eta_ext = (T(1) / pow(T(10), T(0.1))) / pow(bcl_mu_in, T(0.1));
		bcl_eta_in = max2(T(1) / bcl_mu_in, eps_abs);
	}
}

// template <typename T>
// void QPALM_mu_update(
// 		T& primal_feasibility_lhs,
// 		Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_eq_scaled,
// 		Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_in_scaled_l,
// 		Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_eq_scaled_old,
// 		Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_in_scaled_in_old,
// 		T& bcl_eta_ext,
// 		T& bcl_eta_in,
// 		T eps_abs,
// 		isize& n_mu_updates,
// 		Eigen::Matrix<T, Eigen::Dynamic, 1>& mu,
// 		isize dim,
// 		isize n_eq,
// 		isize& n_c,
// 		ldlt::Ldlt<T>& ldl,
// 		qp::QpViewBox<T> qp_scaled,
// 		T rho,
// 		T theta,
// 		T sigmaMax,
// 		T Delta) {
// 	for (isize i = 0; i < n_eq; ++i) {
// 		if (primal_residual_eq_scaled(i) >=
// 		    theta * primal_residual_eq_scaled_old(i)) {
// 			T mu_eq_new = min2(
// 					sigmaMax,
// 					max2(
// 							mu(i) * Delta * primal_residual_eq_scaled(i) /
// 									primal_feasibility_lhs,
// 							mu(i)));

// 			if (n_eq > 0) {
// 				LDLT_MULTI_WORKSPACE_MEMORY(
// 						(e_k_, Init, Vec(dim + n_eq + n_c), LDLT_CACHELINE_BYTES, T));
// 				auto e_k = e_k_.to_eigen().eval();
// 				T diff = T(1) / mu(i) - T(1) / mu_eq_new;
// 				e_k(dim + i) = T(1);
// 				ldl.rank_one_update(e_k, diff);
// 				e_k(dim + i) = T(0);
// 				mu(i) = mu_eq_new;
// 			}
// 		}
// 	}

// 	for (isize i = 0; i < n_c; ++i) {
// 		if (primal_residual_in_scaled_l(i) >=
// 		    theta * primal_residual_in_scaled_in_old(i)) {
// 			T mu_in_new = min2(
// 					sigmaMax,
// 					max2(
// 							mu(n_eq + i) * Delta * primal_residual_in_scaled_l(i) /
// 									primal_feasibility_lhs,
// 							mu(n_eq + i)));

// 			if (n_c > 0) {
// 				LDLT_MULTI_WORKSPACE_MEMORY(
// 						(e_k_, Init, Vec(dim + n_eq + n_c), LDLT_CACHELINE_BYTES, T));
// 				auto e_k = e_k_.to_eigen().eval();
// 				T diff = T(1) / mu(n_eq + i) - T(1) / mu_in_new;
// 				e_k(dim + i) = T(1);
// 				ldl.rank_one_update(e_k, diff);
// 				e_k(dim + i) = T(0);
// 				mu(n_eq + i) = mu_in_new;
// 			}
// 		}
// 	}
// }

template <typename T>
void QPALM_update_fact(
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,
		T eps_abs,
		isize& n_mu_updates,
		T& bcl_mu_in,
		T& bcl_mu_eq,
		VectorViewMut<T> xe,
		VectorViewMut<T> ye,
		VectorViewMut<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,

		isize dim,
		isize n_eq,
		isize n_c,
		ldlt::Ldlt<T>& ldl) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		std::cout << "good step" << std::endl;
		bcl_eta_ext = bcl_eta_ext / T(10);
		bcl_eta_in = max2(bcl_eta_in / T(10), eps_abs);
		ye.to_eigen() = y.to_eigen();
		ze.to_eigen() = z.to_eigen();
		xe.to_eigen() = x.to_eigen();
	} else {
		std::cout << "bad step" << std::endl;
		bcl_eta_in = max2(bcl_eta_in / T(10), eps_abs);
		T new_bcl_mu_in = min2(bcl_mu_in * T(10), T(1e8));
		T new_bcl_mu_eq = min2(bcl_mu_eq * T(10), T(1e10));
		if (bcl_mu_in != new_bcl_mu_in || bcl_mu_eq != new_bcl_mu_eq) {
			{ ++n_mu_updates; }
		}
		qp::detail::mu_update(
				bcl_mu_eq,
				new_bcl_mu_eq,
				bcl_mu_in,
				new_bcl_mu_in,
				dim,
				n_eq,
				n_c,
				ldl);
		bcl_mu_eq = new_bcl_mu_eq;
		bcl_mu_in = new_bcl_mu_in;
	}
}

// COMPUTES:
// primal_residual_eq_scaled = scaled(Ax - b)
//
// primal_feasibility_lhs = max(norm(unscaled(Ax - b)),
//                              norm(unscaled([Cx - u]+ + [Cx - l]-)))
// primal_feasibility_eq_rhs_0 = norm(unscaled(Ax))
// primal_feasibility_in_rhs_0 = norm(unscaled(Cx))
//
// MAY_ALIAS[primal_residual_in_scaled_u, primal_residual_in_scaled_l]
//
// INDETERMINATE:
// primal_residual_in_scaled_u = unscaled(Cx)
// primal_residual_in_scaled_l = unscaled([Cx - u]+ + [Cx - l]-)
template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
void global_primal_residual(
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		VectorViewMut<T> primal_residual_eq_scaled,
		VectorViewMut<T> primal_residual_in_scaled_u,
		VectorViewMut<T> primal_residual_in_scaled_l,
		qp::QpViewBox<T> qp,
		qp::QpViewBox<T> qp_scaled,
		Preconditioner precond,
		VectorView<T> x) {
	LDLT_DECL_SCOPE_TIMER("in solver", "primal residual", T);
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto b_ = qp_scaled.b.to_eigen();

	auto primal_residual_eq_scaled_ = primal_residual_eq_scaled.to_eigen();
	auto primal_residual_in_scaled_u_ = primal_residual_in_scaled_u.to_eigen();
	auto primal_residual_in_scaled_l_ = primal_residual_in_scaled_l.to_eigen();

	primal_residual_eq_scaled_.noalias() = A_ * x_;
	primal_residual_in_scaled_u_.noalias() = C_ * x_;

	precond.unscale_primal_residual_in_place_eq(primal_residual_eq_scaled);
	primal_feasibility_eq_rhs_0 = infty_norm(primal_residual_eq_scaled_);
	precond.unscale_primal_residual_in_place_in(primal_residual_in_scaled_u);
	primal_feasibility_in_rhs_0 = infty_norm(primal_residual_in_scaled_u_);

	primal_residual_eq_scaled_ -= qp.b.to_eigen();
	primal_residual_in_scaled_l_ =
			detail::positive_part(primal_residual_in_scaled_u_ - qp.u.to_eigen()) +
			detail::negative_part(primal_residual_in_scaled_u_ - qp.l.to_eigen());

	primal_feasibility_lhs = max2(
			infty_norm(primal_residual_in_scaled_l_),
			infty_norm(primal_residual_eq_scaled_));
	precond.scale_primal_residual_in_place_eq(primal_residual_eq_scaled);
}

// dual_feasibility_lhs = norm(dual_residual_scaled)
// dual_feasibility_rhs_0 = norm(unscaled(H×x))
// dual_feasibility_rhs_1 = norm(unscaled(AT×y))
// dual_feasibility_rhs_3 = norm(unscaled(CT×z))
//
//
// dual_residual_scaled = scaled(H×x + g + AT×y + CT×z)
// dw_aug = indeterminate
template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
void global_dual_residual(
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3,
		VectorViewMut<T> dual_residual_scaled,
		VectorViewMut<T> dw_aug,
		qp::QpViewBox<T> qp_scaled,
		Preconditioner precond,
		VectorView<T> x,
		VectorView<T> y,
		VectorView<T> z) {
	LDLT_DECL_SCOPE_TIMER("in solver", "dual residual", T);
	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_ = z.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();

	isize const dim = qp_scaled.H.rows;

	auto dual_residual_scaled_ = dual_residual_scaled.to_eigen();
	auto dw_aug_ = dw_aug.to_eigen();

	dual_residual_scaled_ = g_;

	dw_aug_.topRows(dim).setZero();

	dw_aug_.topRows(dim).noalias() = H_ * x_;
	dual_residual_scaled_ += dw_aug_.topRows(dim);
	precond.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, dw_aug_.topRows(dim)});
	dual_feasibility_rhs_0 = infty_norm(dw_aug_.topRows(dim));

	dw_aug_.topRows(dim).setZero();
	dw_aug_.topRows(dim).noalias() = A_.transpose() * y_;
	dual_residual_scaled_ += dw_aug_.topRows(dim);
	precond.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, dw_aug_.topRows(dim)});
	dual_feasibility_rhs_1 = infty_norm(dw_aug_.topRows(dim));

	dw_aug_.topRows(dim).setZero();
	dw_aug_.topRows(dim).noalias() = C_.transpose() * z_;
	dual_residual_scaled_ += dw_aug_.topRows(dim);
	precond.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, dw_aug_.topRows(dim)});
	dual_feasibility_rhs_3 = infty_norm(dw_aug_.topRows(dim));

	precond.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, dual_residual_scaled_});

	dual_feasibility_lhs = infty_norm(dual_residual_scaled_);

	precond.scale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, dual_residual_scaled_});
}

template <typename T>
auto SaddlePointError(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		T mu_eq,
		T mu_in,
		T rho,
		isize n_in) -> T {

	auto H_ = qp_scaled.H.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.as_const().to_eigen();
	auto x_e = xe.to_eigen();
	auto y_ = y.as_const().to_eigen();
	auto y_e = ye.to_eigen();
	auto z_ = z.as_const().to_eigen();
	auto z_e = ze.to_eigen();
	auto b_ = qp_scaled.b.to_eigen();
	auto l_ = qp_scaled.l.to_eigen();
	auto u_ = qp_scaled.u.to_eigen();

	auto prim_in_u = C_ * x_ - u_ - (z_ - z_e) / mu_in;
	auto prim_in_l = C_ * x_ - l_ - (z_ - z_e) / mu_in;

	T prim_eq_e = infty_norm(A_ * x_ - b_ - (y_ - y_e) / mu_eq);
	T dual_e = infty_norm(
			H_ * x_ + rho * (x_ - x_e) + g_ + A_.transpose() * y_ +
			C_.transpose() * z_);
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < n_in; i = i + 1) {
		using std::fabs;
		if (z_(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_u(i)));
		} else if (z_(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(prim_in_u(i), T(0)));
			prim_in_e = max2(prim_in_e, fabs(min2(prim_in_l(i), T(0))));
		}
	}
	err = max2(err, prim_in_e);
	return err;
}

template <typename T>
auto saddle_point(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> y,
		VectorView<T> z,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		T mu_in,
		isize n_in,
		VectorViewMut<T> prim_in_u,
		VectorViewMut<T> prim_in_l,
		VectorView<T> prim_eq,
		VectorViewMut<T> dual_eq) -> T {

	auto H_ = qp_scaled.H.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto x_e = xe.to_eigen();
	auto y_ = y.to_eigen();
	auto y_e = ye.to_eigen();
	auto z_ = z.to_eigen();
	auto z_e = ze.to_eigen();
	auto b_ = qp_scaled.b.to_eigen();
	auto l_ = qp_scaled.l.to_eigen();
	auto u_ = qp_scaled.u.to_eigen();

	prim_in_u.to_eigen() -= z_ / mu_in;
	prim_in_l.to_eigen() -= z_ / mu_in;
	T prim_eq_e = infty_norm(prim_eq.to_eigen());

	dual_eq.to_eigen().noalias() += C_.transpose() * z_;

	T dual_e = infty_norm(dual_eq.to_eigen());
	T err = max2(prim_eq_e, dual_e);

	T prim_in_e(0);

	for (isize i = 0; i < n_in; ++i) {
		using std::fabs;

		if (z_(i) > 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_u(i)));
		} else if (z_(i) < 0) {
			prim_in_e = max2(prim_in_e, fabs(prim_in_l(i)));
		} else {
			prim_in_e = max2(prim_in_e, max2(prim_in_u(i), T(0)));
			prim_in_e = max2(prim_in_e, fabs(min2(prim_in_l(i), T(0))));
		}
	}
	err = max2(err, prim_in_e);
	return err;
}

// z_eq == b
//
// [ H + rho I    AT          CT    ]       [ H×x + g + AT×y_eq + CT×y_in ]
// [ A         -1/µ_eq I      0     ]       [ A×x - z_eq                  ]
// [ C            0        -1/µ_in I] dw = -[ C×x - z_in                  ]

template <typename T>
void newton_step_osqp(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> dw_,
		VectorViewMut<T> err_,
		T mu_eq,
		T mu_in,
		T rho,
		isize dim,
		isize n_eq,
		isize n_in,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<T> rhs_,
		VectorView<T> dual_residual_,     // H×x + g + AT×y_eq + CT×y_in
		VectorView<T> primal_residual_eq_ // A×x-b
) {

	auto rhs = rhs_.to_eigen();
	auto dual_residual = dual_residual_.to_eigen();
	auto primal_residual_eq = primal_residual_eq_.to_eigen();

	auto C_ = qp_scaled.C.to_eigen();
	auto x_e = xe.to_eigen();
	auto z_e = ze.to_eigen();
	auto dw = dw_.to_eigen();
	auto res = err_.to_eigen();
	dw.setZero();
	res.setZero();

	rhs.topRows(dim) = -dual_residual;
	rhs.middleRows(dim, n_eq) = -primal_residual_eq;
	{
		// C×x - z_in
		rhs.tail(n_in) = z_e.tail(n_in);
		rhs.tail(n_in).noalias() -= C_ * x_e;
	}

	detail::iterative_solve_with_permut_fact_osqp(
			rhs_,
			dw_,
			err_,
			ldl,
			T(1e-5),
			isize(10),
			qp_scaled,
			dim,
			n_eq,
			n_in,
			mu_eq,
			mu_in,
			rho);
	dw_.to_eigen() = dw;
}

template <typename T>
void newton_step_fact(
		qp::QpViewBox<T> qp_scaled,
		VectorViewMut<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorViewMut<T> z_pos,
		VectorViewMut<T> z_neg,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c) {

	auto z_pos_ = z_pos.to_eigen();
	auto z_neg_ = z_neg.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();

	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();

	l_active_set_n_u_ = (z_pos_.array() > T(0)).matrix();
	l_active_set_n_l_ = (z_neg_.array() < T(0)).matrix();

	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	isize num_active_inequalities = active_inequalities_.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_err, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));

	auto rhs = _rhs.to_eigen();
	auto dw = _dw.to_eigen();
	auto err = _err.to_eigen();
	{
		qp::line_search::active_set_change_new(
				VectorView<bool>{from_eigen, active_inequalities_},
				current_bijection_map,
				n_c,
				n_in,
				dim,
				n_eq,
				ldl,
				qp_scaled,
				mu_in,
				mu_eq,
				rho);
		rhs.topRows(dim) -= dual_for_eq_;
		for (isize j = 0; j < n_in; ++j) {
			rhs.topRows(dim) -=
					mu_in * (max2(z_pos_(j), T(0)) + min2(z_neg_(j), T(0))) * C_.row(j);
		}
	}
	detail::iterative_solve_with_permut_fact_new( //
			{from_eigen, rhs},
			{from_eigen, dw},
			{from_eigen, err},
			ldl,
			eps,
			isize(5),
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu_eq,
			mu_in,
			rho);

	dx.to_eigen() = dw.topRows(dim);
}

template <typename T>
void newton_step_QPALM(
		qp::QpViewBox<T> qp_scaled,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> dx,
		VectorView<T> mu_,
		T rho,
		T eps,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorView<T> z_pos_,
		VectorView<T> z_neg_,
		VectorView<T> dual_for_eq_,
		VectorViewMut<bool> l_active_set_n_u_,
		VectorViewMut<bool> l_active_set_n_l_,
		VectorViewMut<bool> active_inequalities_,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c) {

	auto H_ = qp_scaled.H.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();

	auto mu = mu_.to_eigen();
	auto z_pos = z_pos_.to_eigen();
	auto z_neg = z_neg_.to_eigen();
	auto dual_for_eq = dual_for_eq_.to_eigen();
	auto l_active_set_n_u = l_active_set_n_u_.to_eigen();
	auto l_active_set_n_l = l_active_set_n_l_.to_eigen();
	auto active_inequalities = active_inequalities_.to_eigen();

	l_active_set_n_u = (z_pos.array() > T(0)).matrix();
	l_active_set_n_l = (z_neg.array() < T(0)).matrix();

	active_inequalities = l_active_set_n_u || l_active_set_n_l;

	isize num_active_inequalities = active_inequalities.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_err, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));

	auto rhs = _rhs.to_eigen().eval();
	auto dw = _dw.to_eigen().eval();
	auto err = _err.to_eigen().eval();
	{
		qp::line_search::active_set_change_QPALM(
				VectorView<bool>{from_eigen, active_inequalities},
				current_bijection_map,
				n_c,
				n_in,
				dim,
				n_eq,
				ldl,
				qp_scaled,
				mu,
				rho);
		rhs.topRows(dim) -= dual_for_eq;
		for (isize j = 0; j < n_in; ++j) {
			rhs.topRows(dim) -= mu(n_eq + j) *
			                    (max2(z_pos(j), T(0)) + min2(z_neg(j), T(0))) *
			                    C_.row(j);
		}
	}

	iterative_solve_with_permut_fact_QPALM( //
			rhs,
			dw,
			err,
			ldl,
			eps,
			isize(5),
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu,
			rho);

	dx.to_eigen() = dw.topRows(dim);
}

template <typename T, typename Preconditioner>
auto initial_guess_fact(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp_scaled,
		T mu_in,
		T mu_eq,
		T rho,
		T eps_int,
		Preconditioner precond,
		isize dim,
		isize n_eq,
		isize n_in,

		VectorViewMut<T> primal_residual_eq,
		VectorViewMut<T> prim_in_u,
		VectorViewMut<T> prim_in_l,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<T> d_dual_for_eq,
		VectorViewMut<T> cdx,
		VectorViewMut<T> d_primal_residual_eq,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		VectorViewMut<T> dw_aug,

		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		T R) -> T {

	auto primal_residual_eq_ = primal_residual_eq.to_eigen();
	auto prim_in_u_ = prim_in_u.to_eigen();
	auto prim_in_l_ = prim_in_l.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto d_dual_for_eq_ = d_dual_for_eq.to_eigen();
	auto cdx_ = cdx.to_eigen();
	auto d_primal_residual_eq_ = d_primal_residual_eq.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();
	auto dw_aug_ = dw_aug.to_eigen();

	auto H_ = qp_scaled.H.to_eigen();
	auto g_ = qp_scaled.g.to_eigen();
	auto A_ = qp_scaled.A.to_eigen();
	auto C_ = qp_scaled.C.to_eigen();
	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_ = z.to_eigen();
	auto z_e = ze.to_eigen().eval();
	auto l_ = qp_scaled.l.to_eigen();
	auto u_ = qp_scaled.u.to_eigen();

	{
		prim_in_u_ = C_ * x_;
		prim_in_l_ = prim_in_u_;

		prim_in_u_ -= u_;
		prim_in_l_ -= l_;
	}

	precond.unscale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_u_});
	precond.unscale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_l_});
	precond.unscale_dual_in_place_in(VectorViewMut<T>{from_eigen, z_e});

	prim_in_u_ += z_e / mu_in;
	prim_in_l_ += z_e / mu_in;

	l_active_set_n_u_.array() = (prim_in_u_.array() >= T(0));
	l_active_set_n_l_.array() = (prim_in_l_.array() <= T(0));

	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	prim_in_u_ -= z_e / mu_in;
	prim_in_l_ -= z_e / mu_in;

	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_u_});
	precond.scale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, prim_in_l_});
	precond.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, z_e});

	// rescale value
	isize num_active_inequalities = active_inequalities_.count();
	isize inner_pb_dim = dim + n_eq + num_active_inequalities;

	qp::line_search::active_set_change_new(
			VectorView<bool>{from_eigen, active_inequalities_},
			current_bijection_map,
			n_c,
			n_in,
			dim,
			n_eq,
			ldl,
			qp_scaled,
			mu_in,
			mu_eq,
			rho);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_rhs, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T),
			(err_it_, Init, Vec(inner_pb_dim), LDLT_CACHELINE_BYTES, T));
	auto rhs = _rhs.to_eigen().eval();
	auto dw = _dw.to_eigen().eval();
	auto err_it = err_it_.to_eigen().eval();

	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			if (l_active_set_n_u_(i)) {
				rhs(j + dim + n_eq) = -prim_in_u_(i);
			} else if (l_active_set_n_l_(i)) {
				rhs(j + dim + n_eq) = -prim_in_l_(i);
			}
		} else {
			rhs.topRows(dim) += z_(i) * C_.row(i);
		}
	}

	rhs.topRows(dim) = -dual_for_eq_;
	rhs.middleRows(dim, n_eq) = -primal_residual_eq_;
	detail::iterative_solve_with_permut_fact_new( //
			{from_eigen, rhs},
			{from_eigen, dw},
			{from_eigen, err_it},
			ldl,
			eps_int,
			isize(5),
			qp_scaled,
			current_bijection_map,
			dim,
			n_eq,
			n_c,
			n_in,
			mu_eq,
			mu_in,
			rho);

	d_dual_for_eq_ = rhs.topRows(dim); // d_dual_for_eq_ = -dual_for_eq_ -C^T dz
	for (isize i = 0; i < n_in; i++) {
		isize j = current_bijection_map(i);
		if (j < n_c) {
			if (l_active_set_n_u_(i)) {
				d_dual_for_eq_ -= dw(j + dim + n_eq) * C_.row(i);
			} else if (l_active_set_n_l_(i)) {
				d_dual_for_eq_ -= dw(j + dim + n_eq) * C_.row(i);
			}
		}
	}

	dw_aug_.setZero();
	dw_aug_.topRows(dim + n_eq) = dw.topRows(dim + n_eq);
	for (isize j = 0; j < n_in; ++j) {
		isize i = current_bijection_map(j);
		if (i < n_c) {
			dw_aug_(j + dim + n_eq) = dw(dim + n_eq + i);
			cdx_(j) = rhs(i + dim + n_eq) + dw(dim + n_eq + i) / mu_in;
		} else {
			dw_aug_(j + dim + n_eq) = -z_(j);
			cdx_(j) = C_.row(j).dot(dw_aug_.topRows(dim));
		}
	}

	prim_in_u_ += z_e / mu_in;
	prim_in_l_ += z_e / mu_in;

	// d_primal_residual_eq_ = (A_ * dw_aug_.topRows(dim) -
	// dw_aug_.middleRows(dim, n_eq) / mu_eq);
	d_primal_residual_eq_ =
			rhs.middleRows(dim, n_eq); // By definition of linear system solution
	// d_dual_for_eq_ = (H_ * dw_aug_.topRows(dim) +  A_.transpose() *
	// dw_aug_.middleRows(dim, n_eq) +  rho * dw_aug_.topRows(dim));

	// cdx_ = C_ * dw_aug_.topRows(dim);
	dual_for_eq_ -= C_.transpose() * z_e;
	T alpha_step = qp::line_search::initial_guess_LS(
			ze,
			{from_eigen, dw_aug_.tail(n_in)},
			{from_eigen, prim_in_l_},
			{from_eigen, prim_in_u_},
			{from_eigen, cdx_},
			{from_eigen, d_dual_for_eq_},
			{from_eigen, dual_for_eq_},
			{from_eigen, d_primal_residual_eq_},
			{from_eigen, primal_residual_eq_},
			qp_scaled.C,
			mu_eq,
			mu_in,
			rho,
			dim,
			n_eq,
			n_in,
			R);

	std::cout << "alpha from initial guess " << alpha_step << std::endl;

	prim_in_u_ += (alpha_step * cdx_);
	prim_in_l_ += (alpha_step * cdx_);
	l_active_set_n_u_ = (prim_in_u_.array() >= T(0)).matrix();
	l_active_set_n_l_ = (prim_in_l_.array() <= T(0)).matrix();
	active_inequalities_ = l_active_set_n_u_ || l_active_set_n_l_;

	x_ += alpha_step * dw_aug_.topRows(dim);
	y_ += alpha_step * dw_aug_.middleRows(dim, n_eq);

	for (isize i = 0; i < n_in; ++i) {
		if (l_active_set_n_u_(i)) {
			z(i) = max2(z(i) + alpha_step * dw_aug_(dim + n_eq + i), T(0));
		} else if (l_active_set_n_l_(i)) {
			z(i) = min2(z(i) + alpha_step * dw_aug_(dim + n_eq + i), T(0));
		} else {
			z(i) += alpha_step * dw_aug_(dim + n_eq + i);
		}
	}
	primal_residual_eq_ += alpha_step * d_primal_residual_eq_;
	dual_for_eq_ += alpha_step * d_dual_for_eq_;

	T err = detail::saddle_point(
			qp_scaled,
			x.as_const(),
			y.as_const(),
			z.as_const(),
			xe,
			ye,
			ze,
			mu_in,
			n_in,
			{from_eigen, prim_in_u_},
			{from_eigen, prim_in_l_},
			{from_eigen, primal_residual_eq_},
			{from_eigen, dual_for_eq_});
	return err;
}

template <typename T>
auto correction_guess(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp_scaled,
		T mu_in,
		T mu_eq,
		T rho,
		T eps_int,
		isize dim,
		isize n_eq,
		isize n_in,
		isize max_iter_in,
		isize& n_tot,

		VectorViewMut<T> residual_in_y,
		VectorViewMut<T> z_pos,
		VectorViewMut<T> z_neg,
		VectorViewMut<T> dual_for_eq,
		VectorViewMut<T> Hdx,
		VectorViewMut<T> Adx,
		VectorViewMut<T> Cdx,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,

		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		VectorViewMut<T> dw_aug,
		T& correction_guess_rhs_g) -> T {

	auto residual_in_y_ = residual_in_y.to_eigen();
	auto z_pos_ = z_pos.to_eigen();
	auto z_neg_ = z_neg.to_eigen();
	auto dual_for_eq_ = dual_for_eq.to_eigen();
	auto Hdx_ = Hdx.to_eigen();
	auto Adx_ = Adx.to_eigen();
	auto Cdx_ = Cdx.to_eigen();
	auto l_active_set_n_u_ = l_active_set_n_u.to_eigen();
	auto l_active_set_n_l_ = l_active_set_n_l.to_eigen();
	auto active_inequalities_ = active_inequalities.to_eigen();
	auto dw_aug_ = dw_aug.to_eigen();

	T err_in = T(0);

	for (i64 iter = 0; iter <= max_iter_in; ++iter) {

		if (iter == max_iter_in) {
			n_tot += max_iter_in;
			break;
		}

		dw_aug_.topRows(dim).setZero();

		qp::detail::newton_step_fact(
				qp_scaled,
				VectorViewMut<T>{from_eigen, dw_aug_.topRows(dim)},
				mu_eq,
				mu_in,
				rho,
				eps_int,
				dim,
				n_eq,
				n_in,
				{from_eigen, z_pos_},
				{from_eigen, z_neg_},
				{from_eigen, dual_for_eq_},
				{from_eigen, l_active_set_n_u_},
				{from_eigen, l_active_set_n_l_},
				{from_eigen, active_inequalities_},

				ldl,
				current_bijection_map,
				n_c);
		T alpha_step = T(1);
		Hdx_ = (qp_scaled.H).to_eigen() * dw_aug_.topRows(dim);
		Adx_ = (qp_scaled.A).to_eigen() * dw_aug_.topRows(dim);
		Cdx_ = (qp_scaled.C).to_eigen() * dw_aug_.topRows(dim);
		if (n_in > isize(0)) {
			alpha_step = qp::line_search::correction_guess_LS(
					{from_eigen, Hdx_},
					{from_eigen, Adx_},
					{from_eigen, Cdx_},
					{from_eigen, residual_in_y_},
					{from_eigen, z_pos_},
					{from_eigen, z_neg_},
					{from_eigen, dw_aug_.topRows(dim)},
					qp_scaled.g,
					x.as_const(),
					xe,
					ye,
					ze,
					mu_eq,
					mu_in,
					rho,
					n_in

			);
		}

		if (infty_norm(alpha_step * dw_aug_.topRows(dim)) < T(1.E-11)) {
			n_tot += iter + 1;
			break;
		}

		x.to_eigen() += alpha_step * dw_aug_.topRows(dim);
		z_pos_ += alpha_step * Cdx_;
		z_neg_ += alpha_step * Cdx_;
		residual_in_y_ += alpha_step * Adx_;
		y.to_eigen() = mu_eq * residual_in_y_;
		dual_for_eq_ +=
				alpha_step * (mu_eq * (qp_scaled.A).to_eigen().transpose() * Adx_ +
		                  rho * dw_aug_.topRows(dim) + Hdx_);
		for (isize j = 0; j < n_in; ++j) {
			z(j) = mu_in * (max2(z_pos_(j), T(0)) + min2(z_neg_(j), T(0)));
		}

		Hdx_ = (qp_scaled.H).to_eigen() * x.to_eigen();
		T rhs_c = max2(correction_guess_rhs_g, infty_norm(Hdx_));

		dw_aug_.topRows(dim) =
				(qp_scaled.A.to_eigen().transpose()) * (y.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug_.topRows(dim)));
		Hdx_ += (dw_aug_.topRows(dim));

		dw_aug_.topRows(dim) =
				(qp_scaled.C.to_eigen().transpose()) * (z.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug_.topRows(dim)));
		Hdx_ += (dw_aug_.topRows(dim));

		Hdx_ += ((qp_scaled.g).to_eigen());

		err_in = infty_norm(Hdx_ + rho * (x.to_eigen() - xe.to_eigen()));
		std::cout << "---it in " << iter << " projection norm " << err_in
							<< " alpha " << alpha_step << std::endl;

		if (err_in <= eps_int * (1 + rhs_c)) {
			n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}

template <typename T>
T correction_guess_QPALM(
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBoxMut<T> qp_scaled,
		VectorView<T> mu_,
		T rho,
		T eps_int,
		isize dim,
		isize n_eq,
		isize n_in,
		isize max_iter_in,
		isize& n_tot,
		VectorView<T> residual_in_y_,
		VectorView<T> z_pos_,
		VectorView<T> z_neg_,
		VectorView<T> dual_for_eq_,
		VectorView<T> Hdx_,
		VectorView<T> Adx_,
		VectorView<T> Cdx_,
		VectorViewMut<bool> l_active_set_n_u,
		VectorViewMut<bool> l_active_set_n_l,
		VectorViewMut<bool> active_inequalities,
		ldlt::Ldlt<T>& ldl,
		VectorViewMut<isize> current_bijection_map,
		isize& n_c,
		VectorViewMut<T> dw_aug_,
		T& correction_guess_rhs_g) {

	auto mu = mu_.to_eigen();
	auto residual_in_y = residual_in_y_.to_eigen();
	auto z_pos = z_pos_.to_eigen();
	auto z_neg = z_neg_.to_eigen();
	auto dual_for_eq = dual_for_eq_.to_eigen();
	auto Hdx = Hdx_.to_eigen();
	auto Adx = Adx_.to_eigen();
	auto Cdx = Cdx_.to_eigen();
	auto dw_aug = dw_aug_.to_eigen();

	T err_in = T(0);

	for (i64 iter = 0; iter <= max_iter_in; ++iter) {

		if (iter == max_iter_in) {
			n_tot += max_iter_in;
			break;
		}

		dw_aug.topRows(dim).setZero();

		qp::detail::newton_step_QPALM<T>(
				qp_scaled.as_const(),
				x.as_const(),
				xe,
				ye,
				ze,
				VectorViewMut<T>{from_eigen, dw_aug.topRows(dim)},
				mu,
				rho,
				eps_int,
				dim,
				n_eq,
				n_in,
				z_pos_,
				z_neg_,
				dual_for_eq_,
				l_active_set_n_u,
				l_active_set_n_l,
				active_inequalities,

				ldl,
				current_bijection_map,
				n_c);
		T alpha_step = T(1);

		Hdx = (qp_scaled.H).to_eigen() * dw_aug.topRows(dim);
		Adx = (qp_scaled.A).to_eigen() * dw_aug.topRows(dim);
		Cdx = (qp_scaled.C).to_eigen() * dw_aug.topRows(dim);

		if (n_in > isize(0)) {
			alpha_step = qp::line_search::correction_guess_LS_QPALM(
					Hdx,
					VectorView<T>{from_eigen, dw_aug.topRows(dim)},
					(qp_scaled.g).as_const(),
					Adx,
					Cdx,
					residual_in_y,
					z_pos,
					z_neg,
					x.as_const(),
					xe,
					ye,
					ze,
					mu,
					rho,
					n_in,
					n_eq);
		}

		if (infty_norm(alpha_step * dw_aug.topRows(dim)) < T(1.E-11)) {
			n_tot += iter + 1;
			break;
		}

		x.to_eigen() += alpha_step * dw_aug.topRows(dim);
		z_pos += alpha_step * Cdx;
		z_neg += alpha_step * Cdx;
		residual_in_y += alpha_step * Adx;
		y.to_eigen().array() = mu.topRows(n_eq).array() * residual_in_y.array();
		dual_for_eq.array() +=
				(alpha_step * mu.topRows(n_eq).array() *
		     (qp_scaled.A.to_eigen().transpose() * Adx).array()) +
				(rho * dw_aug.topRows(dim) + Hdx).array();
		for (isize j = 0; j < n_in; ++j) {
			z(j) = mu(n_eq + j) * (max2(z_pos(j), T(0)) + min2(z_neg(j), T(0)));
		}

		Hdx = (qp_scaled.H).to_eigen() * x.to_eigen();
		T rhs_c = max2(correction_guess_rhs_g, infty_norm(Hdx));

		dw_aug.topRows(dim) = (qp_scaled.A.to_eigen().transpose()) * (y.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug.topRows(dim)));
		Hdx += dw_aug.topRows(dim);

		dw_aug.topRows(dim) = (qp_scaled.C.to_eigen().transpose()) * (z.to_eigen());
		rhs_c = max2(rhs_c, infty_norm(dw_aug.topRows(dim)));
		Hdx += dw_aug.topRows(dim);

		Hdx += (qp_scaled.g).to_eigen();

		err_in = infty_norm(Hdx + rho * (x.to_eigen() - xe.to_eigen()));
		std::cout << "---it in " << iter << " projection norm " << err_in
							<< " alpha " << alpha_step << std::endl;

		if (err_in <= eps_int * (1 + rhs_c)) {
			n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats qpSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		T err_IG,
		T beta,
		T R,
		Preconditioner precond = Preconditioner{}) {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_tot_max = dim + n_eq + n_in;
	isize n_c = 0;

	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto bcl_mu_eq = T(1e3);
	auto bcl_mu_in = T(1e1);
	T bcl_eta_ext = 1 / pow(bcl_mu_in, T(0.1));
	T bcl_eta_in = T(1);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_h_ws, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_g_scaled, Uninit, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_a_scaled, Uninit, Mat(n_eq, dim), LDLT_CACHELINE_BYTES, T),
			(_c_scaled, Uninit, Mat(n_in, dim), LDLT_CACHELINE_BYTES, T),
			(_b_scaled, Uninit, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_u_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_l_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_residual_scaled, Init, Vec(n_tot_max + n_in), LDLT_CACHELINE_BYTES, T),
			(_ye, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_ze, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_xe, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_in, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_kkt,
	     Uninit,
	     Mat(dim + n_eq + n_c, dim + n_eq + n_c),
	     LDLT_CACHELINE_BYTES,
	     T),

			(_current_bijection_map, Init, Vec(n_in), LDLT_CACHELINE_BYTES, isize),

			(_dw_aug, Init, Vec(n_tot_max), LDLT_CACHELINE_BYTES, T),
			(_d_dual_for_eq, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_cdx, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_d_primal_residual_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_l_active_set_n_u, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(_l_active_set_n_l, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(_active_inequalities, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));

	auto d_dual_for_eq = _d_dual_for_eq.to_eigen();
	auto Cdx = _cdx.to_eigen();
	auto d_primal_residual_eq = _d_primal_residual_eq.to_eigen();
	auto l_active_set_n_u = _l_active_set_n_u.to_eigen();
	auto l_active_set_n_l = _l_active_set_n_l.to_eigen();
	auto active_inequalities = _active_inequalities.to_eigen();
	auto dw_aug = _dw_aug.to_eigen();

	auto current_bijection_map = _current_bijection_map.to_eigen();
	for (isize i = 0; i < n_in; i++) {
		current_bijection_map(i) = i;
	}

	auto H_copy = _h_scaled.to_eigen();
	auto kkt = _kkt.to_eigen();
	auto H_ws = _h_ws.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto C_copy = _c_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
	auto u_copy = _u_scaled.to_eigen();
	auto l_copy = _l_scaled.to_eigen();

	auto ye = _ye.to_eigen();
	auto ze = _ze.to_eigen();
	auto xe = _xe.to_eigen();
	auto diag_diff_in = _diag_diff_in.to_eigen();
	auto diag_diff_eq = _diag_diff_eq.to_eigen();

	auto residual_scaled = _residual_scaled.to_eigen();

	H_copy = qp.H.to_eigen();
	q_copy = qp.g.to_eigen();
	A_copy = qp.A.to_eigen();
	b_copy = qp.b.to_eigen();
	C_copy = qp.C.to_eigen();
	u_copy = qp.u.to_eigen();
	l_copy = qp.l.to_eigen();
	auto qp_scaled = qp::QpViewBoxMut<T>{
			{from_eigen, H_copy},
			{from_eigen, q_copy},
			{from_eigen, A_copy},
			{from_eigen, b_copy},
			{from_eigen, C_copy},
			{from_eigen, u_copy},
			{from_eigen, l_copy}};

	// LDLT_DECL_SCOPE_TIMER("in solver", "scaling", T);
	precond.scale_qp_in_place(qp_scaled);

	// LDLT_DECL_SCOPE_TIMER("in solver", "setting H", T);
	H_ws = H_copy;
	for (isize i = 0; i < dim; ++i) {
		H_ws(i, i) += rho;
	}
	kkt.topLeftCorner(dim, dim) = H_ws;
	kkt.block(0, dim, dim, n_eq) = qp_scaled.A.to_eigen().transpose();
	kkt.block(dim, 0, n_eq, dim) = qp_scaled.A.to_eigen();
	kkt.bottomRightCorner(n_eq + n_c, n_eq + n_c).setZero();
	kkt.diagonal().segment(dim, n_eq).setConstant(-T(1) / bcl_mu_eq);
	ldlt::Ldlt<T> ldl{decompose, kkt};

	// LDLT_DECL_SCOPE_TIMER("in solver", "warm starting", T);
	ldlt::Ldlt<T> ldl_ws{decompose, H_ws};
	x.to_eigen() = -qp_scaled.g.to_eigen();
	ldl_ws.solve_in_place(x.to_eigen());

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	T correction_guess_rhs_g = infty_norm(qp_scaled.g.to_eigen());

	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u =
			residual_scaled.middleRows(dim + n_eq, n_in);
	auto primal_residual_in_scaled_l = residual_scaled.bottomRows(n_in);

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);

	T refactor_dual_feasibility_threshold = T(1e-2);
	T refactor_rho_threshold = T(1e-7);
	T refactor_rho_update_factor = T(10);

	T cold_reset_bcl_mu_max = T(1e8);
	T cold_reset_primal_test_factor = T(1);
	T cold_reset_dual_test_factor = T(1);
	T cold_reset_mu_eq = T(1.1);
	T cold_reset_mu_in = T(1.1);

	for (i64 iter = 0; iter <= max_iter; ++iter) {
		if (iter == max_iter) {
			break;
		}
		n_ext += 1;

		// compute primal residual

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);
		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());
		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw_aug},
				qp_scaled.as_const(),
				precond,
				x.as_const(),
				y.as_const(),
				z.as_const());

		std::cout << "---------------it : " << iter
							<< " primal residual : " << primal_feasibility_lhs
							<< " dual residual : " << dual_feasibility_lhs << std::endl;
		std::cout << "bcl_eta_ext : " << bcl_eta_ext
							<< " bcl_eta_in : " << bcl_eta_in << " rho : " << rho
							<< " bcl_mu_eq : " << bcl_mu_eq << " bcl_mu_in : " << bcl_mu_in
							<< std::endl;

		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(eps_abs +
		     eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 primal_feasibility_rhs_1_eq,
												 primal_feasibility_rhs_1_in_u),
										 primal_feasibility_rhs_1_in_l)

										 ));

		bool is_dual_feasible =
				dual_feasibility_lhs <=
				(eps_abs +
		     eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 dual_feasibility_rhs_2)));

		if (is_primal_feasible) {
			if (dual_feasibility_lhs > refactor_dual_feasibility_threshold && //
			    rho > refactor_rho_threshold) {
				T rho_new = max2( //
						T(rho / refactor_rho_update_factor),
						refactor_rho_threshold);

				qp::detail::refactorize(
						qp_scaled.as_const(),
						VectorViewMut<isize>{from_eigen, current_bijection_map},
						MatrixViewMut<T, colmajor>{from_eigen, kkt},
						n_c,
						bcl_mu_in,
						rho,
						rho_new,
						ldl);

				rho = rho_new;
			}

			if (is_dual_feasible) {

				LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
				precond.unscale_primal_in_place(x);
				precond.unscale_dual_in_place_eq(y);
				precond.unscale_dual_in_place_in(z);

				return {n_ext, n_mu_updates, n_tot};
			}
		}

		xe = x.to_eigen();
		ye = y.to_eigen();
		ze = z.to_eigen();

		bool do_initial_guess_fact = primal_feasibility_lhs < err_IG;

		T err_in(0.);

		if (do_initial_guess_fact) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "initial guess", T);
			err_in = qp::detail::initial_guess_fact(
					VectorView<T>{from_eigen, xe},
					VectorView<T>{from_eigen, ye},
					VectorView<T>{from_eigen, ze},
					x,
					y,
					z,
					qp_scaled.as_const(),
					bcl_mu_in,
					bcl_mu_eq,
					rho,
					bcl_eta_in,
					precond,
					dim,
					n_eq,
					n_in,
					{from_eigen, primal_residual_eq_scaled},
					{from_eigen, primal_residual_in_scaled_u},
					{from_eigen, primal_residual_in_scaled_l},
					{from_eigen, dual_residual_scaled},
					{from_eigen, d_dual_for_eq},
					{from_eigen, Cdx},
					{from_eigen, d_primal_residual_eq},
					{from_eigen, l_active_set_n_u},
					{from_eigen, l_active_set_n_l},
					{from_eigen, active_inequalities},
					{from_eigen, dw_aug},

					ldl,
					VectorViewMut<isize>{from_eigen, current_bijection_map},
					n_c,
					R);

			n_tot += 1;
		}

		bool do_correction_guess = !do_initial_guess_fact ||
		                           (do_initial_guess_fact && err_in >= bcl_eta_in);

		if (do_correction_guess) {
			auto CT_z = qp_scaled.C.trans().to_eigen() * z.to_eigen();
			auto AT_primal_res =
					qp_scaled.A.trans().to_eigen() * primal_residual_eq_scaled;
			dual_residual_scaled.noalias() -= CT_z;
			dual_residual_scaled.noalias() += bcl_mu_eq * AT_primal_res;
		}

		if (do_initial_guess_fact && err_in >= bcl_eta_in) {
			primal_residual_eq_scaled += y.to_eigen() / bcl_mu_eq;

			primal_residual_in_scaled_u += z.to_eigen() / bcl_mu_in;
			primal_residual_in_scaled_l += z.to_eigen() / bcl_mu_in;
		}
		if (!do_initial_guess_fact) {
			auto Cx = qp_scaled.C.to_eigen() * x.to_eigen();

			primal_residual_eq_scaled += ye / bcl_mu_eq;
			primal_residual_in_scaled_u = ze / bcl_mu_in;

			primal_residual_in_scaled_u.noalias() += Cx;

			primal_residual_in_scaled_l = primal_residual_in_scaled_u;

			primal_residual_in_scaled_u -= qp_scaled.u.to_eigen();
			primal_residual_in_scaled_l -= qp_scaled.l.to_eigen();
		}

		if (do_correction_guess) {

			// LDLT_DECL_SCOPE_TIMER("in solver", "correction guess", T);
			err_in = qp::detail::correction_guess(
					{from_eigen, xe},
					{from_eigen, ye},
					{from_eigen, ze},
					x,
					y,
					z,
					qp_scaled.as_const(),
					bcl_mu_in,
					bcl_mu_eq,
					rho,
					bcl_eta_in,
					dim,
					n_eq,
					n_in,
					max_iter_in,
					n_tot,
					{from_eigen, primal_residual_eq_scaled},
					{from_eigen, primal_residual_in_scaled_u},
					{from_eigen, primal_residual_in_scaled_l},
					{from_eigen, dual_residual_scaled},
					{from_eigen, d_dual_for_eq},
					{from_eigen, d_primal_residual_eq},
					{from_eigen, Cdx},
					{from_eigen, l_active_set_n_u},
					{from_eigen, l_active_set_n_l},
					{from_eigen, active_inequalities},

					ldl,
					VectorViewMut<isize>{from_eigen, current_bijection_map},
					n_c,
					{from_eigen, dw_aug},
					correction_guess_rhs_g);

			std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
								<< " error from initial guess : " << err_in << " bcl_eta_in "
								<< bcl_eta_in << std::endl;
		}

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);
		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::detail::global_primal_residual(
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());

		// LDLT_DECL_SCOPE_TIMER("in solver", "BCL", T);
		qp::detail::BCL_update_fact(
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				eps_abs,
				n_mu_updates,
				bcl_mu_in,
				bcl_mu_eq,
				VectorViewMut<T>{from_eigen, ye},
				VectorViewMut<T>{from_eigen, ze},
				y,
				z,
				dim,
				n_eq,
				n_c,
				ldl,
				beta);

		// COLD RESTART

		// LDLT_DECL_SCOPE_TIMER("in solver", "residuals", T);

		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		qp::detail::global_dual_residual(
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw_aug},
				qp_scaled.as_const(),
				precond,
				x.as_const(),
				y.as_const(),
				z.as_const());

		if ((primal_feasibility_lhs_new >=
		     cold_reset_primal_test_factor *
		         max2(primal_feasibility_lhs, machine_eps)) &&

		    (dual_feasibility_lhs_new >=
		     cold_reset_dual_test_factor *
		         max2(primal_feasibility_lhs, machine_eps)) &&

		    max2(bcl_mu_eq, bcl_mu_in) >= cold_reset_bcl_mu_max) {
			std::cout << "cold restart" << std::endl;

			T new_bcl_mu_eq = cold_reset_mu_eq;
			T new_bcl_mu_in = cold_reset_mu_in;

			qp::detail::mu_update(
					bcl_mu_eq,
					new_bcl_mu_eq,
					bcl_mu_in,
					new_bcl_mu_in,
					dim,
					n_eq,
					n_c,
					ldl);

			bcl_mu_in = new_bcl_mu_in;
			bcl_mu_eq = new_bcl_mu_eq;
		}
	}

	return {max_iter, n_mu_updates, n_tot};
}

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
QpSolveStats QPALMSolve( //
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		T eps_IG,
		T R,
		Preconditioner precond = Preconditioner{}) {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_c = 0;

	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto bcl_mu_eq = T(1e3);
	auto bcl_mu_in = T(1e1);
	T bcl_eta_ext = T(1);
	T bcl_eta_in = T(1);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_h_ws, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_g_scaled, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_a_scaled, Uninit, Mat(n_eq, dim), LDLT_CACHELINE_BYTES, T),
			(_c_scaled, Uninit, Mat(n_in, dim), LDLT_CACHELINE_BYTES, T),
			(_b_scaled, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_u_scaled, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_l_scaled, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_residual_scaled, Init, Vec(dim + n_eq + n_in), LDLT_CACHELINE_BYTES, T),
			(_y, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_z, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(xe_, Init, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_eq, Init, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_diag_diff_in, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),

			(_kkt,
	     Uninit,
	     Mat(dim + n_eq + n_c, dim + n_eq + n_c),
	     LDLT_CACHELINE_BYTES,
	     T),

			(current_bijection_map_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, isize),

			(_dw_aug, Init, Vec(dim + n_eq + n_in), LDLT_CACHELINE_BYTES, T),
			(d_dual_for_eq_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_cdx, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(d_primal_residual_eq_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(l_active_set_n_u_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(l_active_set_n_l_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool),
			(active_inequalities_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));

	auto d_dual_for_eq = d_dual_for_eq_.to_eigen();
	auto Cdx = _cdx.to_eigen();
	auto d_primal_residual_eq = d_primal_residual_eq_.to_eigen();
	auto l_active_set_n_u = l_active_set_n_u_.to_eigen();
	auto l_active_set_n_l = l_active_set_n_l_.to_eigen();
	auto active_inequalities = active_inequalities_.to_eigen();
	auto dw_aug = _dw_aug.to_eigen();

	auto current_bijection_map = current_bijection_map_.to_eigen();
	for (isize i = 0; i < n_in; i++) {
		current_bijection_map(i) = i;
	}

	auto H_copy = _h_scaled.to_eigen();
	auto kkt = _kkt.to_eigen();
	auto H_ws = _h_ws.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto C_copy = _c_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
	auto u_copy = _u_scaled.to_eigen();
	auto l_copy = _l_scaled.to_eigen();

	H_copy = qp.H.to_eigen();
	q_copy = qp.g.to_eigen();
	A_copy = qp.A.to_eigen();
	b_copy = qp.b.to_eigen();
	C_copy = qp.C.to_eigen();
	u_copy = qp.u.to_eigen();
	l_copy = qp.l.to_eigen();

	auto qp_scaled = qp::QpViewBoxMut<T>{
			{from_eigen, H_copy},
			{from_eigen, q_copy},
			{from_eigen, A_copy},
			{from_eigen, b_copy},
			{from_eigen, C_copy},
			{from_eigen, u_copy},
			{from_eigen, l_copy}};
	precond.scale_qp_in_place(qp_scaled);

	kkt.topLeftCorner(dim, dim) = qp_scaled.H.to_eigen();
	for (isize i = 0; i < dim; ++i) {
		kkt(i, i) += rho;
	}
	kkt.block(0, dim, dim, n_eq) = qp_scaled.A.to_eigen().transpose();
	kkt.block(dim, 0, n_eq, dim) = qp_scaled.A.to_eigen();
	kkt.bottomRightCorner(n_eq + n_c, n_eq + n_c).setZero();
	{
		T tmp_eq = -T(1) / bcl_mu_eq;
		T tmp_in = -T(1) / bcl_mu_in;
		for (isize i = 0; i < n_eq; ++i) {
			kkt(dim + i, dim + i) = tmp_eq;
		}
		for (isize i = 0; i < n_c; ++i) {
			kkt(dim + n_eq + i, dim + n_eq + i) = tmp_in;
		}
	}

	ldlt::Ldlt<T> ldl{decompose, kkt};

	H_ws = H_copy;
	for (isize i = 0; i < dim; ++i) {
		H_ws(i, i) += rho;
	}

	ldlt::Ldlt<T> ldl_ws{decompose, H_ws};
	x.to_eigen() = -(qp_scaled.g).to_eigen();
	ldl_ws.solve_in_place(x.to_eigen());

	auto residual_scaled = _residual_scaled.to_eigen();

	auto ye = _y.to_eigen();
	auto ze = _z.to_eigen();
	auto xe = xe_.to_eigen();
	auto diag_diff_in = _diag_diff_in.to_eigen();
	auto diag_diff_eq = _diag_diff_eq.to_eigen();

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	T correction_guess_rhs_g = infty_norm((qp_scaled.g).to_eigen());

	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u = residual_scaled.bottomRows(n_in);
	auto primal_residual_in_scaled_l = residual_scaled.bottomRows(n_in);

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);

	xe = x.to_eigen().eval();
	ye = y.to_eigen().eval();
	ze = z.to_eigen().eval();
	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext += 1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());

		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw_aug},
				qp_scaled.as_const(),
				precond,
				x.as_const(),
				y.as_const(),
				z.as_const());

		std::cout << "---------------it : " << iter
							<< " primal residual : " << primal_feasibility_lhs
							<< " dual residual : " << dual_feasibility_lhs << std::endl;
		std::cout << "bcl_eta_ext : " << bcl_eta_ext
							<< " bcl_eta_in : " << bcl_eta_in << " rho : " << rho
							<< " bcl_mu_eq : " << bcl_mu_eq << " bcl_mu_in : " << bcl_mu_in
							<< std::endl;
		bool is_primal_feasible =
				primal_feasibility_lhs <=
				(eps_abs +
		     eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 primal_feasibility_rhs_1_eq,
												 primal_feasibility_rhs_1_in_u),
										 primal_feasibility_rhs_1_in_l)

										 ));

		bool is_dual_feasible =
				dual_feasibility_lhs <=
				(eps_abs +
		     eps_rel * max2(
											 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
											 max2( //
													 dual_feasibility_rhs_1,
													 dual_feasibility_rhs_2)));

		if (is_primal_feasible) {

			if (is_dual_feasible) {
				{
					LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
					precond.unscale_primal_in_place(x);
					precond.unscale_dual_in_place_eq(y);
					precond.unscale_dual_in_place_in(z);
				}
				return {n_ext, n_mu_updates, n_tot};
			}
		}

		primal_residual_eq_scaled += (ye / bcl_mu_eq);
		dual_residual_scaled +=
				(rho * (x.to_eigen() - xe) -
		     (qp_scaled.A).to_eigen().transpose() * y.to_eigen() -
		     (qp_scaled.C).to_eigen().transpose() * z.to_eigen() +
		     bcl_mu_eq * (qp_scaled.A).to_eigen().transpose() *
		         primal_residual_eq_scaled);
		primal_residual_in_scaled_u = qp_scaled.C.to_eigen() * x.to_eigen() -
		                              qp_scaled.u.to_eigen() + ze / bcl_mu_in;
		primal_residual_in_scaled_l = qp_scaled.C.to_eigen() * x.to_eigen() -
		                              qp_scaled.l.to_eigen() + ze / bcl_mu_in;

		T err_in = qp::detail::correction_guess(
				{from_eigen, xe},
				{from_eigen, ye},
				{from_eigen, ze},
				x,
				y,
				z,
				qp_scaled.as_const(),
				bcl_mu_in,
				bcl_mu_eq,
				rho,
				bcl_eta_in,
				dim,
				n_eq,
				n_in,
				max_iter_in,
				n_tot,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				{from_eigen, dual_residual_scaled},
				{from_eigen, d_primal_residual_eq},
				{from_eigen, Cdx},
				{from_eigen, d_dual_for_eq},
				{from_eigen, l_active_set_n_u},
				{from_eigen, l_active_set_n_l},
				{from_eigen, active_inequalities},

				ldl,
				VectorViewMut<isize>{from_eigen, current_bijection_map},
				n_c,
				{from_eigen, dw_aug},
				correction_guess_rhs_g);
		std::cout << "primal_feasibility_lhs " << primal_feasibility_lhs
							<< " error from inner loop : " << err_in << " bcl_eta_in "
							<< bcl_eta_in << std::endl;

		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::detail::global_primal_residual(
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_l},
				qp,
				qp_scaled.as_const(),
				precond,
				x.as_const());

		qp::detail::QPALM_update_fact(
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				eps_abs,
				n_mu_updates,
				bcl_mu_in,
				bcl_mu_eq,
				VectorViewMut<T>{from_eigen, xe},
				VectorViewMut<T>{from_eigen, ye},
				VectorViewMut<T>{from_eigen, ze},
				x,
				y,
				z,
				dim,
				n_eq,
				n_c,
				ldl);
		/*
		qp::detail::QPALM_mu_update(
		      T& primal_feasibility_lhs,
		      Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_eq_scaled,
		      Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_in_scaled_l,
		      Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_eq_scaled_old,
		      Eigen::Matrix<T, Eigen::Dynamic, 1>& primal_residual_in_scaled_in_old,
		      T& bcl_eta_ext,
		      T& bcl_eta_in,
		      T eps_abs,
		      isize& n_mu_updates,
		      Eigen::Matrix<T, Eigen::Dynamic, 1>& mu,
		      isize dim,
		      isize n_eq,
		      isize& n_c,
		      ldlt::Ldlt<T>& ldl,
		      qp::QpViewBox<T> qp_scaled,
		      T rho,
		      T theta,
		      T sigmaMax,
		      T Delta
		);
		*/
	}

	return {max_iter, n_mu_updates, n_tot};
}

template <
		typename T,
		typename Preconditioner = qp::preconditioner::IdentityPrecond>
auto osqpSolve( //
		VectorViewMut<T> xe,
		VectorViewMut<T> ye,
		qp::QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		Preconditioner precond = Preconditioner{}) -> QpSolveStats {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;
	isize const max_n_tot = dim + n_eq + n_in;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto mu_eq = T(1e4);
	auto mu_in = T(1e1);
	T alpha = T(1.6);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled, Uninit, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_htot, Uninit, Mat(max_n_tot, max_n_tot), LDLT_CACHELINE_BYTES, T),
			(_g_scaled, Uninit, Vec(dim), LDLT_CACHELINE_BYTES, T),
			(_b_scaled, Uninit, Vec(n_eq), LDLT_CACHELINE_BYTES, T),
			(_u_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_l_scaled, Uninit, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(_residual_scaled, Init, Vec(max_n_tot), LDLT_CACHELINE_BYTES, T),
			(_ze, Init, Vec(n_eq + n_in), LDLT_CACHELINE_BYTES, T),
			(_z, Init, Vec(n_eq + n_in), LDLT_CACHELINE_BYTES, T),
			(_dw, Init, Vec(max_n_tot), LDLT_CACHELINE_BYTES, T),
			(_rhs, Init, Vec(max_n_tot), LDLT_CACHELINE_BYTES, T),
			(_err, Init, Vec(max_n_tot), LDLT_CACHELINE_BYTES, T),
			(_tmp, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T));

	auto dw = _dw.to_eigen();
	auto err = _err.to_eigen();
	auto tmp = _tmp.to_eigen();

	auto q_copy = _g_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
	auto u_copy = _u_scaled.to_eigen();
	auto l_copy = _l_scaled.to_eigen();

	auto residual_scaled = _residual_scaled.to_eigen();

	q_copy = qp.g.to_eigen();
	b_copy = qp.b.to_eigen();
	u_copy = qp.u.to_eigen();
	l_copy = qp.l.to_eigen();

	auto Htot = _htot.to_eigen();
	auto rhs = _rhs.to_eigen();

	Htot.bottomRightCorner(n_eq + n_in, n_eq + n_in).setZero();

	Htot.topLeftCorner(dim, dim) = qp.H.to_eigen();

	// only set bottom left half
	Htot.block(dim, 0, n_eq, dim) = qp.A.to_eigen();
	Htot.block(dim + n_eq, 0, n_in, dim) = qp.C.to_eigen();

	auto qp_scaled = [&] {
		auto qp_scaled_mut = qp::QpViewBoxMut<T>{
				{from_eigen, Htot.topLeftCorner(dim, dim)},
				{from_eigen, q_copy},
				{from_eigen, Htot.block(dim, 0, n_eq, dim)},
				{from_eigen, b_copy},
				{from_eigen, Htot.block(dim + n_eq, 0, n_in, dim)},
				{from_eigen, u_copy},
				{from_eigen, l_copy}};
		precond.scale_qp_in_place(qp_scaled_mut);
		return qp_scaled_mut.as_const();
	}();

	{
		// update diagonal H part
		for (isize i = 0; i < dim; ++i) {
			Htot(i, i) += rho;
		}

		// update diagonal constraint part
		T tmp_eq = -T(1) / mu_eq;
		T tmp_in = -T(1) / mu_in;
		for (isize i = 0; i < n_eq; ++i) {
			Htot(dim + i, dim + i) = tmp_eq;
		}
		for (isize i = 0; i < n_in; ++i) {
			Htot(dim + n_eq + i, dim + n_eq + i) = tmp_in;
		}
	}

	{
		// initial primal guess
		ldlt::Ldlt<T> ldl_ws{decompose, Htot.topLeftCorner(dim, dim)};
		xe.to_eigen() = -(qp_scaled.g.to_eigen());
		ldl_ws.solve_in_place(xe.to_eigen());
	}

	ldlt::Ldlt<T> ldl{decompose, Htot};

	auto ze = _ze.to_eigen();
	ze.topRows(n_eq) = qp_scaled.b.to_eigen();
	auto z = _z.to_eigen();
	z.topRows(n_eq) = qp_scaled.b.to_eigen();

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u = residual_scaled.bottomRows(n_in);

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	T rhs_d(0);
	T rhs_p(0);
	T fact(0);

	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext += 1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::global_primal_residual(
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				{from_eigen, primal_residual_eq_scaled},
				{from_eigen, primal_residual_in_scaled_u},
				{from_eigen, primal_residual_in_scaled_u},
				qp,
				qp_scaled,
				precond,
				xe.as_const());

		qp::detail::global_dual_residual(
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3,
				{from_eigen, dual_residual_scaled},
				{from_eigen, dw},
				qp_scaled,
				precond,
				xe.as_const(),
				ye.as_const().segment(0, n_eq),
				ye.as_const().segment(n_eq, n_in));

		std::cout << "---------------it : " << iter
							<< " primal residual : " << primal_feasibility_lhs
							<< " dual residual : " << dual_feasibility_lhs << std::endl;
		std::cout << " rho : " << rho << " mu_eq : " << mu_eq
							<< " mu_in : " << mu_in << std::endl;

		rhs_d = max_list({
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_rhs_1_eq,
				primal_feasibility_rhs_1_in_u,
				primal_feasibility_rhs_1_in_l,
		});

		rhs_p = max_list({
				dual_feasibility_rhs_3,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_2,
		});

		bool is_primal_feasible =
				primal_feasibility_lhs <= (eps_abs + eps_rel * rhs_p);

		bool is_dual_feasible = dual_feasibility_lhs <= (eps_abs + eps_rel * rhs_d);

		if (is_primal_feasible) {
			if (is_dual_feasible) {

				// POLISHING IFF IT HAS CONVERGED
				rhs.topRows(dim) = -dual_residual_scaled;
				rhs.middleRows(dim, n_eq) = -primal_residual_eq_scaled;
				{
					LDLT_MULTI_WORKSPACE_MEMORY(
							(test_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));
					auto test = test_.to_eigen();
					isize j(0);
					for (isize i = 0; i < n_in; ++i) {
						test(i) = qp_scaled.u.to_eigen()(i) - ze(i) >= ye.to_eigen()(i) &&
						          ze(i) - qp_scaled.l.to_eigen()(i) >= -ye.to_eigen()(i);
						if (test(i)) {
							ldl.delete_at(j + dim + n_eq);
						} else {
							rhs.topRows(dim) += qp_scaled.C.to_eigen().row(i) * ze(i);
							rhs(dim + n_eq + j) =
									-(qp_scaled.C.to_eigen().row(i).dot(xe.to_eigen()) - ze(i));
							j += 1;
						}
					}
					LDLT_MULTI_WORKSPACE_MEMORY(
							(dw___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T),
							(err___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T),
							(rhs___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T));
					dw.setZero();
					err.topRows(dim + n_eq + j).setZero();

					auto dw__ = dw___.to_eigen();
					auto err__ = err___.to_eigen();
					auto rhs__ = rhs___.to_eigen();

					dw__ = dw.topRows(dim + n_eq + j);
					err__ = err.topRows(dim + n_eq + j);
					rhs__ = rhs.topRows(dim + n_eq + j);

					iterative_solve_with_permut_fact_osqp(
							rhs___,
							dw___,
							err___,
							ldl,
							T(1e-5),
							isize(3),
							qp_scaled,
							dim,
							n_eq,
							j,
							mu_eq,
							mu_in,
							rho);
					j = isize(0);
					for (isize i = 0; i < n_in; ++i) {
						if (test(i)) {
							dw(dim + n_eq + i) = dw__(dim + n_eq + j);
							j += 1;
						}
					}
				}

				// see end of loop for comments
				tmp = (alpha / mu_in) * dw.tail(n_in) //
				      + ze.tail(n_in)                 //
				      + ye.to_eigen().tail(n_in) / mu_in;

				z.tail(n_in) = tmp + //
				               detail::positive_part(qp_scaled.l.to_eigen() - tmp) -
				               detail::positive_part(tmp - qp_scaled.u.to_eigen());

				ye.to_eigen().topRows(n_eq) += alpha * dw;
				ye.to_eigen().tail(n_in) = mu_in * (tmp - z.tail(n_in));
				xe.to_eigen() += alpha * dw.topRows(dim);

				// unscale polished solution
				{
					LDLT_DECL_SCOPE_TIMER("in solver", "unscale solution", T);
					precond.unscale_primal_in_place(xe);
					precond.unscale_dual_in_place_eq(
							VectorViewMut<T>{from_eigen, ye.to_eigen().topRows(n_eq)});
					precond.unscale_dual_in_place_in(
							VectorViewMut<T>{from_eigen, ye.to_eigen().tail(n_in)});
				}

				return {n_ext, n_mu_updates, n_tot};
			}
		}

		// mu update

		if (iter > 1) {
			using std::sqrt;
			fact = sqrt(
					(primal_feasibility_lhs * rhs_d) /
					(dual_feasibility_lhs * rhs_p + machine_eps));
			if (fact > T(5) || fact < T(0.2)) {
				T mu_in_new = min2(mu_in * fact, T(1e6));
				T mu_eq_new = min2(mu_eq * fact, T(1e6));
				if (mu_in_new != T(1.e6) || mu_eq_new != T(1e6)) {
					qp::detail::mu_update(
							mu_eq, mu_eq_new, mu_in, mu_in_new, dim, n_eq, n_in, ldl);
				}

				mu_in = mu_in_new;
				mu_eq = mu_eq_new;
				n_mu_updates += 1;
			}
		}

		// NEWTON STEP

		qp::detail::newton_step_osqp(
				qp_scaled,
				xe.as_const(),
				ye.as_const(),
				VectorView<T>{from_eigen, ze},
				VectorViewMut<T>{from_eigen, dw},
				VectorViewMut<T>{from_eigen, err},
				mu_eq,
				mu_in,
				rho,
				dim,
				n_eq,
				n_in,
				ldl,
				VectorViewMut<T>{from_eigen, rhs},
				VectorView<T>{from_eigen, dual_residual_scaled},
				VectorView<T>{from_eigen, primal_residual_eq_scaled});

		// ITERATES UPDATES according to OSQP algorithm 1 page 9 using

		// tmp = alpha/µ dw + zk + yk/µ
		tmp = (alpha / mu_in) * dw.tail(n_in) //
		      + ze.tail(n_in)                 //
		      + ye.to_eigen().tail(n_in) / mu_in;

		z.tail(n_in) = tmp + //
		               detail::positive_part(qp_scaled.l.to_eigen() - tmp) -
		               detail::positive_part(tmp - qp_scaled.u.to_eigen());

		// y{k+1} = yk + µ ( alpha (zk + dw/µ) + (1-alpha) zk - z{k+1} )
		//        = yk + µ ( zk - z{k+1} + alpha/µ dw )
		//        = yk + µ (zk - z{k+1}) + alpha dw
		//
		// eq constraints: z_k == z_{k+1} == b
		ye.to_eigen().topRows(n_eq) += alpha * dw;
		ye.to_eigen().tail(n_in) = mu_in * (tmp - z.tail(n_in));
		xe.to_eigen() += alpha * dw.topRows(dim);

		ze = z;
	}

	return {max_iter, n_mu_updates, n_tot};
}

} // namespace detail
} // namespace qp

#endif /* end of include guard INRIA_LDLT_EQ_SOLVER_HPP_HDWGZKCLS */
