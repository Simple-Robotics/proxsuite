#ifndef INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS
#define INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS

#include "ldlt/views.hpp"
#include "qp/views.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <list>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}
namespace line_search {

template <typename T>
struct LineSearchResult {
	T grad;
	T a0;
	T b0;
};

struct ActiveSetChangeResult {
	Eigen::Matrix<isize, Eigen::Dynamic, 1> new_bijection_map;
	isize n_c_f;
};

template <typename T, Layout LC>
auto gradient_norm_computation_box(
		VectorView<T> ze,
		Eigen::Matrix<T, Eigen::Dynamic, 1>& dz,
		T mu_in,
		MatrixView<T, LC> C,
		VectorView<T> Cdx_,
		VectorView<T> residual_in_z_u_,
		VectorView<T> residual_in_z_l_,
		VectorView<T> d_dual_for_eq_,
		VectorView<T> dual_for_eq_,
		VectorView<T> d_primal_residual_eq_,
		VectorView<T> primal_residual_eq_,
		T alpha,
		isize dim,
		isize n_eq,
		isize n_in) -> T {

	/*
	 * Compute the squared norm of the following vector res
	 *
	 * vect1 = H.dot(x) + g + rho_primal_proximal * (x-xe)
	 *       + A.transpose().dot(y)
	 *       + C[active_inequalities_l,:].T × z[active_inequalities_l]
	 *       + C[active_inequalities_u,:].T × z[active_inequalities_u]
	 *
	 * vect3_u = residual_in_u[active_inequalities_u]
	 *         - (z[active_inequalities_u] - ze[active_inequalities_u])/mu_in
	 * vect3_l = residual_in_l[active_inequalities_l]
	 *         - (z[active_inequalities_l]-ze[active_inequalities_l])/mu_in
	 * vect4 = z[inactive_inequalities]
	 *
	 * res = np.concatenate((
	 *         vect1,
	 *         (residual_eq * mu_eq - (y-ye))/mu_eq,
	 *         vect3_u,
	 *         vect3_l,
	 *         vect4,
	 *       ), axis = None)
	 *
	 * considering the following qp problem : (H, g, A, b, C, u,l) and
	 *
	 * residual_eq = A.dot(x) - b
	 * residual_in_u = C.dot(x) - u
	 * residual_in_l = C.dot(x) - l
	 * active_inequalities_u = residual_in_u + z/mu_in >= 0
	 * active_inequalities_u = residual_in_l + z/mu_in <= 0
	 * active_inequalities = active_inequalities_u + active_inequalities_l
	 * inactive_inequalities = ~active_inequalities
	 */

	auto C_copy = C.to_eigen();
	auto z_e = ze.to_eigen();

	auto Cdx = Cdx_.to_eigen();
	auto residual_in_z_u = residual_in_z_u_.to_eigen();
	auto residual_in_z_l = residual_in_z_l_.to_eigen();
	auto d_dual_for_eq = d_dual_for_eq_.to_eigen();
	auto dual_for_eq = dual_for_eq_.to_eigen();
	auto d_primal_residual_eq = d_primal_residual_eq_.to_eigen();
	auto primal_residual_eq = primal_residual_eq_.to_eigen();

	// define active set
	auto tmp_u = residual_in_z_u + alpha * Cdx;
	auto tmp_l = residual_in_z_l + alpha * Cdx;

	// TODO: buggy?
	// auto active_set_tmp_u = (tmp_u).array() >= 0 ;
	// auto active_set_tmp_l = (tmp_l).array() <= 0 ;
	// auto inactive_set_tmp = !tmp_u && !tmp_l;
	// auto num_inactive = inactive_set_tmp.count() ;
	// auto num_active_u = active_set_tmp_u.count();
	// auto num_active_l = active_set_tmp_l.count();

	isize num_active_u = 0;
	isize num_active_l = 0;
	isize num_inactive = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) >= T(0.)) {
			num_active_u += 1;
		}
		if (tmp_l(k) <= T(0.)) {
			num_active_l += 1;
		}
		if (tmp_u(k) < T(0.) && tmp_l(k) > T(0.)) {
			num_inactive += 1;
		}
	}

	LDLT_MULTI_WORKSPACE_MEMORY(
			((_active_set_u, Init, Vec(num_active_u), LDLT_CACHELINE_BYTES, isize), //
	     (_inactive_set, Init, Vec(num_inactive), LDLT_CACHELINE_BYTES, isize), //
	     (_active_set_l, Init, Vec(num_active_l), LDLT_CACHELINE_BYTES, isize)));

	auto active_set_u = _active_set_u.to_eigen();
	auto active_set_l = _active_set_l.to_eigen();
	auto inactive_set = _inactive_set.to_eigen();

	active_set_u.setZero();
	isize i_u = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) >= T(0.)) {
			active_set_u(i_u) = k;
			i_u += 1;
		}
	}
	active_set_l.setZero();
	isize i_l = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_l(k) <= T(0.)) {
			active_set_l(i_l) = k;
			i_l += 1;
		}
	}

	isize i_inact = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) < T(0.) && tmp_l(k) > T(0.)) {
			inactive_set(i_inact) = k;
			i_inact += 1;
		}
	}

	// form the gradient
	Eigen::Matrix<T, Eigen::Dynamic, 1> active_part_z(n_in);

	active_part_z = z_e + alpha * dz;
	for (isize k = 0; k < num_active_u; k = k + 1) {
		if (active_part_z(active_set_u(k)) < T(0.)) {
			active_part_z(active_set_u(k)) = T(0.);
		}
	}
	for (isize k = 0; k < num_active_l; k = k + 1) {

		if (active_part_z(active_set_l(k)) > T(0.)) {
			active_part_z(active_set_l(k)) = T(0.);
		}
	}
	Eigen::Matrix<T, Eigen::Dynamic, 1> res(dim + n_eq + n_in);
	res.setZero();

	res.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq;
	Eigen::Matrix<T, Eigen::Dynamic, 1> aux_u(dim);
	aux_u.setZero();
	for (isize k = 0; k < num_active_u; ++k) {
		res.topRows(dim) +=
				active_part_z(active_set_u(k)) * C_copy.row(active_set_u(k));
		aux_u += active_part_z(active_set_u(k)) * C_copy.row(active_set_u(k));
	}
	Eigen::Matrix<T, Eigen::Dynamic, 1> aux_l(dim);
	aux_l.setZero();
	for (isize k = 0; k < num_active_l; ++k) {
		res.topRows(dim) +=
				active_part_z(active_set_l(k)) * C_copy.row(active_set_l(k));
		aux_l += active_part_z(active_set_l(k)) * C_copy.row(active_set_l(k));
	}
	res.middleRows(dim, n_eq) = primal_residual_eq + alpha * d_primal_residual_eq;
	for (isize k = 0; k < num_active_u; ++k) {
		res(dim + n_eq + k) =
				tmp_u(active_set_u(k)) - active_part_z(active_set_u(k)) / mu_in;
	}
	for (isize k = 0; k < num_active_l; ++k) {
		res(dim + n_eq + num_active_u + k) =
				tmp_l(active_set_l(k)) - active_part_z(active_set_l(k)) / mu_in;
	}
	for (isize k = 0; k < num_inactive; ++k) {
		res(dim + n_eq + num_active_u + num_active_l + k) =
				active_part_z(inactive_set(k));
	}

	return res.squaredNorm();
}

template <typename T>
auto gradient_norm_qpalm_box(
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		T alpha,
		VectorView<T> Hdx_,
		VectorView<T> g_,
		VectorView<T> Adx_,
		VectorView<T> residual_in_y_,
		VectorView<T> residual_in_z_u_,
		VectorView<T> residual_in_z_l_,
		VectorView<T> Cdx_,
		isize n_in) -> T {

	/*
	 * the function computes the first derivative of the proximal augmented
	 * lagrangian of the problem
	 *
	 * phi(alpha) = f(x_k+alpha dx) + rho/2 |x_k + alpha dx - x_k|**2
	 *              + mu_eq/2 (|A(x_k+alpha dx)-d+y_k/mu_eq|**2 - |y_k/mu_eq|**2)
	 *              + mu_in/2 ( | [C(x_k+alpha dx) - u + z_k/mu_in]_+ |**2
	 *                         +| [C(x_k+alpha dx) - l + z_k/mu_in]_- |**2
	 *                         - |z_k / mu_in|**2 )
	 * with f(x) = 0.5 * x^THx + g^Tx
	 * phi is a second order polynomial in alpha.
	 * Below are computed its coefficient a0 and b0
	 * in order to compute the desired gradient a0 * alpha + b0
	 */

	auto x_ = x.to_eigen();
	auto xe_ = xe.to_eigen();
	auto dx_ = dx.to_eigen();

	auto Cdx = Cdx_.to_eigen();
	auto Hdx = Hdx_.to_eigen();
	auto Adx = Adx_.to_eigen();
	auto g = g_.to_eigen();
	auto residual_in_y = residual_in_y_.to_eigen();
	auto residual_in_z_u = residual_in_z_u_.to_eigen();
	auto residual_in_z_l = residual_in_z_l_.to_eigen();

	// define active set
	auto tmp_u = residual_in_z_u + Cdx * alpha;
	auto tmp_l = residual_in_z_l + Cdx * alpha;
	auto active_set_tmp_u = tmp_u.array() > 0;
	auto active_set_tmp_l = tmp_l.array() < 0;

	// TODO: buggy
	// auto num_active_u = active_set_tmp_u.count();
	// auto num_active_l = active_set_tmp_l.count();

	isize num_active_u = 0;
	isize num_active_l = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (active_set_tmp_u(k)) {
			num_active_u += 1;
		}
		if (active_set_tmp_l(k)) {
			num_active_l += 1;
		}
	}

	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_u(num_active_u);
	active_set_u.setZero();
	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_l(num_active_l);
	active_set_l.setZero();
	isize i = 0;
	isize j = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (active_set_tmp_u(k)) {
			active_set_u(i) = k;
			i += 1;
		}
		if (active_set_tmp_l(k)) {
			active_set_l(j) = k;
			j += 1;
		}
	}

	// coefficient computation

	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_a0_u(num_active_u);
	tmp_a0_u.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_b0_u(num_active_u);
	tmp_b0_u.setZero();

	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_a0_l(num_active_l);
	tmp_a0_l.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_b0_l(num_active_l);
	tmp_b0_l.setZero();

	for (isize k = 0; k < num_active_u; k = k + 1) {
		tmp_a0_u(k) = Cdx(active_set_u(k));
		tmp_b0_u(k) = residual_in_z_u(active_set_u(k));
	}
	for (isize k = 0; k < num_active_l; k = k + 1) {
		tmp_a0_l(k) = Cdx(active_set_l(k));
		tmp_b0_l(k) = residual_in_z_l(active_set_l(k));
	}
	for (isize k = 0; k < num_active_l; k = k + 1) {
		tmp_a0_l(k) = Cdx(active_set_l(k));
		tmp_b0_l(k) = residual_in_z_l(active_set_l(k));
	}

	T a = dx_.dot(Hdx) + mu_eq * (Adx).squaredNorm() +
	      mu_in * (tmp_a0_u.squaredNorm() + tmp_a0_l.squaredNorm()) +
	      rho * dx_.squaredNorm();

	T b = x_.dot(Hdx) + (rho * (x_ - xe_) + g).dot(dx_) +
	      mu_eq * (Adx).dot(residual_in_y) +
	      mu_in * (tmp_a0_l.dot(tmp_b0_l) + tmp_a0_u.dot(tmp_b0_u));

	return a * alpha + b;
}

template <typename T, Layout LC>
auto local_saddle_point_box(
		VectorView<T> ze,
		VectorView<T> dz_,
		T mu_in,
		MatrixView<T, LC> C,
		VectorView<T> Cdx_,
		VectorView<T> residual_in_z_u_,
		VectorView<T> residual_in_z_l_,
		VectorView<T> d_dual_for_eq_,
		VectorView<T> dual_for_eq_,
		VectorView<T> d_primal_residual_eq_,
		VectorView<T> primal_residual_eq_,
		T alpha,
		isize n_in) -> T {
	/*
	 * the function returns the unique minimum of the positive second order
	 * polynomial in alpha of the L2 norm of the following vector:
	 * concat((
	 *   H.dot(x) + g + rho_primal_proximal * (x-xe)
	 *   + A.transpose() × y
	 *   + C[active_inequalities_l,:].T × z[active_inequalities_l]
	 *   + C[active_inequalities_u,:].T × z[active_inequalities_u],
	 *   residual_eq - (y-ye)/mu_eq,
	 *
	 *   residual_in_u[active_inequalities_u]
	 *   - (z[active_inequalities_u]-ze[active_inequalities_u])/mu_in,
	 *
	 *   residual_in_l[active_inequalities_l]
	 *   - (z[active_inequalities_l]-ze[active_inequalities_l])/mu_in,
	 *
	 *   z[inactive_inequalities],
	 * ))
	 *
	 * with
	 * x = xe + alpha dx
	 * y = ye + alpha dy
	 * z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u], 0)
	 * z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l], 0)
	 *
	 * Furthermore
	 * residual_eq = A.dot(x) - b
	 * residual_in_u = C.dot(x) - u
	 * residual_in_l = C.dot(x) - l
	 * active_inequalities_u = residual_in_u + alpha Cdx >=0
	 * active_inequalities_l = residual_in_l + alpha Cdx <=0
	 * active_inequalities = active_inequalities_u + active_inequalities_l
	 * inactive_inequalities = ~active_inequalities
	 *
	 * To do so the L2 norm is expanded and the exact coefficients of the
	 * polynomial a0 alpha**2 + b0 alpha + c0 are derived.
	 * The argmin is then equal to -b0/2a0 if a0 != 0 and is changed INPLACE
	 * (erasing then alpha entry)
	 * the function returns the L2 norm of the merit function evaluated at the
	 * argmin value found
	 */

	auto C_copy = C.to_eigen();
	auto z_e = ze.to_eigen();
	auto dz = dz_.to_eigen();

	auto Cdx = Cdx_.to_eigen();
	auto residual_in_z_u = residual_in_z_u_.to_eigen();
	auto residual_in_z_l = residual_in_z_l_.to_eigen();
	auto d_dual_for_eq = d_dual_for_eq_.to_eigen().eval();
	auto dual_for_eq = dual_for_eq_.to_eigen().eval();
	auto d_primal_residual_eq = d_primal_residual_eq_.to_eigen();
	auto primal_residual_eq = primal_residual_eq_.to_eigen();

	auto tmp_u = residual_in_z_u + alpha * Cdx;
	auto tmp_l = residual_in_z_l + alpha * Cdx;

	isize num_active_u = 0;
	isize num_active_l = 0;
	isize num_inactive = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) >= T(0.)) {
			num_active_u += 1;
		}
		if (tmp_l(k) <= T(0.)) {
			num_active_l += 1;
		}
		if (tmp_u(k) < T(0.) && tmp_l(k) > T(0.)) {
			num_inactive += 1;
		}
	}

	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_u(num_active_u);
	active_set_u.setZero();
	isize i_u = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) >= T(0.)) {
			active_set_u(i_u) = k;
			i_u += 1;
		}
	}
	Eigen::Matrix<isize, Eigen::Dynamic, 1> active_set_l(num_active_l);
	active_set_l.setZero();
	isize i_l = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_l(k) <= T(0.)) {
			active_set_l(i_l) = k;
			i_l += 1;
		}
	}

	Eigen::Matrix<isize, Eigen::Dynamic, 1> inactive_set(num_inactive);
	isize i_inact = 0;
	for (isize k = 0; k < n_in; k = k + 1) {
		if (tmp_u(k) < T(0.) && tmp_l(k) > T(0.)) {
			inactive_set(i_inact) = k;
			i_inact += 1;
		}
	}

	// form the gradient
	Eigen::Matrix<T, Eigen::Dynamic, 1> z_p(n_in);
	z_p = z_e;
	Eigen::Matrix<T, Eigen::Dynamic, 1> dz_p(n_in);
	dz_p = dz;

	for (isize k = 0; k < num_active_u; ++k) {
		T test = z_e(active_set_u(k)) + alpha * dz(active_set_u(k));
		if (test < 0) {
			z_p(active_set_u(k)) = 0;
			dz_p(active_set_u(k)) = 0;
		}
	}
	for (isize k = 0; k < num_active_l; ++k) {
		T test2 = z_e(active_set_l(k)) + alpha * dz(active_set_l(k));
		if (test2 > 0) {
			z_p(active_set_l(k)) = 0;
			dz_p(active_set_l(k)) = 0;
		}
	}

	// a0 computation

	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d2_u(num_active_u);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d2_l(num_active_l);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp_d3(num_inactive);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp2_u(num_active_u);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp2_l(num_active_l);
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp3(num_inactive);

	for (isize k = 0; k < num_active_u; ++k) {
		d_dual_for_eq += dz_p(active_set_u(k)) * C_copy.row(active_set_u(k));
		tmp_d2_u(k) = Cdx(active_set_u(k)) - dz_p(active_set_u(k)) / mu_in;
	}
	for (isize k = 0; k < num_active_l; ++k) {
		d_dual_for_eq += dz_p(active_set_l(k)) * C_copy.row(active_set_l(k));
		tmp_d2_l(k) = Cdx(active_set_l(k)) - dz_p(active_set_l(k)) / mu_in;
	}

	for (isize k = 0; k < num_inactive; ++k) {
		tmp_d3(k) = dz_p(inactive_set(k));
	}
	T a0 = d_dual_for_eq.squaredNorm() + tmp_d2_u.squaredNorm() +
	       tmp_d2_l.squaredNorm() + tmp_d3.squaredNorm() +
	       d_primal_residual_eq.squaredNorm();

	// b0 computation
	for (isize k = 0; k < num_active_u; ++k) {
		dual_for_eq += z_p(active_set_u(k)) * C_copy.row(active_set_u(k));
		tmp2_u(k) = residual_in_z_u(active_set_u(k)) - z_p(active_set_u(k)) / mu_in;
	}
	for (isize k = 0; k < num_active_l; ++k) {
		dual_for_eq += z_p(active_set_l(k)) * C_copy.row(active_set_l(k));
		tmp2_l(k) = residual_in_z_l(active_set_l(k)) - z_p(active_set_l(k)) / mu_in;
	}
	for (isize k = 0; k < num_inactive; ++k) {
		tmp3(k) = z_p(inactive_set(k));
	}

	T b0 = 2 * (d_dual_for_eq.dot(dual_for_eq) + tmp_d2_u.dot(tmp2_u) +
	            tmp_d2_l.dot(tmp2_l) + tmp3.dot(tmp_d3) +
	            primal_residual_eq.dot(d_primal_residual_eq));

	// c0 computation
	T c0 = dual_for_eq.squaredNorm() + tmp2_u.squaredNorm() + tmp3.squaredNorm() +
	       tmp2_l.squaredNorm() + primal_residual_eq.squaredNorm();

	// derivation of the loss function value and corresponding argmin alpha

	auto res = T(0);

	if (a0 != 0) {
		alpha = (-b0 / (2 * a0));
		res = a0 * pow(alpha, T(2)) + b0 * alpha + c0;
	} else if (b0 != 0) {
		alpha = (-c0 / (b0));
		res = b0 * alpha + c0;
	} else {
		alpha = 0;
		res = c0;
	}

	return res;
}

template <typename T>
auto initial_guess_line_search_box(
		VectorView<T> x,
		VectorView<T> y,
		VectorView<T> ze,
		VectorView<T> dw,
		T mu_eq,
		T mu_in,
		T rho,
		qp::QpViewBox<T> qp) -> T {
	/*
	 * Considering the following qp = (H, g, A, b, C, u,l) and a Newton step
	 * (dx,dy,dz) the fonction gives one optimal alpha minimizing the L2 norm
	 * of the following vector
	 * concat((
	 *   H.dot(x) + g + rho_primal_proximal * (x-xe)
	 *   + A.transpose() × y
	 *   + C[active_inequalities_l,:].T × z[active_inequalities_l]
	 *   + C[active_inequalities_u,:].T × z[active_inequalities_u],
	 *   residual_eq - (y-ye)/mu_eq,
	 *
	 *   residual_in_u[active_inequalities_u]
	 *   - (z[active_inequalities_u]-ze[active_inequalities_u])/mu_in,
	 *
	 *   residual_in_l[active_inequalities_l]
	 *   - (z[active_inequalities_l]-ze[active_inequalities_l])/mu_in,
	 *
	 *   z[inactive_inequalities],
	 * ))
	 *
	 * with
	 * x = xe + alpha dx
	 * y = ye + alpha dy
	 * z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u], 0)
	 * z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l], 0)
	 *
	 * Furthermore
	 * residual_eq = A.dot(x) - b
	 * residual_in_u = C.dot(x) - u
	 * residual_in_l = C.dot(x) - l
	 * active_inequalities_u = residual_in_u + alpha Cdx >=0
	 * active_inequalities_l = residual_in_l + alpha Cdx <=0
	 * active_inequalities = active_inequalities_u + active_inequalities_l
	 * inactive_inequalities = ~active_inequalities
	 *
	 * It can be shown that when one optimal active set is found for the qp
	 * problem, then the optimal alpha canceling (hence minimizing) the L2 norm
	 * of the merit function is unique and equal to 1
	 *
	 * If the optimal active set is not found, one optimal alpha found can not
	 * deviate new iterates formed from the sub problem solution
	 * To do so the algorithm has the following structure :
	 * 1/
	 * 1.1/ it computes the "nodes" alpha which cancel
	 * C.dot(xe+alpha dx) - u, C.dot(xe+alpha dx) - l and ze + alpha dz
	 *
	 * 1.2/ it prepares all needed algebra in order not to derive it each time
	 * (TODO : integrate it at a higher level in the solver)
	 * 2/
	 * 2.1/ it sorts the alpha nodes
	 *
	 * 2.2/ for each "node" it derives the L2 norm of the vector to minimize
	 * (see function: gradient_norm_computation_box) and stores it
	 *
	 * 3/ it defines all intervals on which the active set is constant
	 * 3.1/ it  define intervals (for ex with n+1 nodes):
	 * [alpha[0]-1;alpha[0]],[alpha[0],alpha[1]], ....; [alpha[n],alpha[n]+1]]
	 *
	 * 3.2/ for each interval
	 * it derives the mean node (alpha[i]+alpha[i+1])/2 and the corresponding
	 * active sets active_inequalities_u and active_inequalities_
	 * cap ze and d
	 *
	 * optimal lagrange multiplier z satisfy
	 * z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u], 0)
	 * z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l], 0
	 *
	 * 3.3/ on this interval the merit function is a second order polynomial in
	 * alpha
	 * the function "local_saddle_point_box" derives the exact minimum and
	 * corresponding merif function L2 norm (for this minimum
	 *
	 * 3.4/ if the argmin is within the interval [alpha[i],alpha[i+1]] is
	 * stores the argmin and corresponding L2 norm
	 *
	 * 4/ if the list of argmin obtained from intervals is not empty the
	 * algorithm return the one minimizing the most the merit functio
	 * Otherwise, it returns the node minimizing the most the merit function
	 */

	T machine_eps = std::numeric_limits<T>::epsilon();
	T machine_inf = std::numeric_limits<T>::infinity();

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;

	auto H = (qp.H).to_eigen();
	auto A = (qp.A).to_eigen();
	auto b = (qp.b).to_eigen();
	auto C = (qp.C).to_eigen();
	auto u = (qp.u).to_eigen();
	auto l = (qp.l).to_eigen();

	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_e = ze.to_eigen();
	auto dx_ = dw.to_eigen().head(dim);
	auto dy_ = dw.to_eigen().middleRows(dim, n_eq);
	Eigen::Matrix<T, Eigen::Dynamic, 1> dz_ = dw.to_eigen().tail(n_in);

	T alpha = 1;

	/////////// STEP 1 ////////////
	// computing the "nodes" alphas which cancel  C.dot(xe+alpha dx) - u,
	// C.dot(xe+alpha dx) - l and ze + alpha dz  /////////////

	std::list<T> alphas = {}; // TODO use a vector instead of a list
	// 1.1 add solutions of equation z+alpha dz = 0

	for (isize i = 0; i < n_in; i++) {
		if (std::abs(z_e(i)) != 0) {
			alphas.push_back(-z_e(i) / (dz_(i) + machine_eps));
		}
	}

	// 1.1 add solutions of equations C(x+alpha dx)-u +ze/mu_in = 0 and C(x+alpha
	// dx)-l +ze/mu_in = 0

	Eigen::Matrix<T, Eigen::Dynamic, 1> Cdx = C * dx_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_in_z_u =
			C * x_ - u + z_e / mu_in;
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_in_z_l =
			C * x_ - l + z_e / mu_in;

	for (isize i = 0; i < n_in; i++) {
		if (std::abs(Cdx(i)) != 0) {
			alphas.push_back(-residual_in_z_u(i) / (Cdx(i) + machine_eps));
			alphas.push_back(-residual_in_z_l(i) / (Cdx(i) + machine_eps));
		}
	}

	// 1.2 it prepares all needed algebra in order not to derive it each time
	// (TODO : integrate it at a higher level in the solver)

	Eigen::Matrix<T, Eigen::Dynamic, 1> g = (qp.g).to_eigen();
	Eigen::Matrix<T, Eigen::Dynamic, 1> dual_for_eq =
			H * x_ + g + A.transpose() * y_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> d_dual_for_eq =
			H * dx_ + A.transpose() * dy_ + rho * dx_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> d_primal_residual_eq =
			A * dx_ - dy_ / mu_eq;
	Eigen::Matrix<T, Eigen::Dynamic, 1> primal_residual_eq = A * x_ - b;

	if (alphas.empty() == false) {
		//////// STEP 2 ////////
		// 2.1/ it sort alpha nodes
		alphas.sort();
		alphas.unique();

		// 2.2/ for each node active set and associated gradient is computed

		std::list<T> liste_norm_grad_noeud = {};

		for (auto a : alphas) {

			if (std::abs(a) < T(1.e6)) {

				// calcul de la norm du gradient du noeud
				T grad_norm = line_search::gradient_norm_computation_box(
						ze,
						dz_,
						mu_in,
						qp.C,
						{from_eigen, Cdx},
						{from_eigen, residual_in_z_u},
						{from_eigen, residual_in_z_l},
						{from_eigen, d_dual_for_eq},
						{from_eigen, dual_for_eq},
						{from_eigen, d_primal_residual_eq},
						{from_eigen, primal_residual_eq},
						a,
						dim,
						n_eq,
						n_in);
				liste_norm_grad_noeud.push_back(grad_norm);
			} else {
				liste_norm_grad_noeud.push_back(machine_inf);
			}
		}

		//////////STEP 3 ////////////
		// 3.1 : define intervals with alphas

		std::list<T> liste_norm_grad_interval = {};
		std::list<T> liste_argmin = {};

		std::list<T> interval = alphas;
		interval.push_front((alphas.front() - T(1)));
		interval.push_back((alphas.back() + T(1)));

		std::vector<T> intervals{std::begin(interval), std::end(interval)};
		isize n_ = isize(intervals.size());

		for (isize i = 0; i < n_ - 1; ++i) {

			// 3.2 : it derives the mean node (alpha[i]+alpha[i+1])/2
			// the corresponding active sets active_inequalities_u and
			// active_inequalities_l cap ze and dz is derived through function
			// local_saddle_point_box

			T a_ = (intervals[usize(i)] + intervals[usize(i + 1)]) / T(2.0);

			// 3.3 on this interval the merit function is a second order
			// polynomial in alpha
			// the function "local_saddle_point_box" derives the exact minimum
			// and corresponding merit function L2 norm (for this minimum)
			T associated_grad_2_norm = line_search::local_saddle_point_box(
					ze,
					{from_eigen, dz_},
					mu_in,
					qp.C,
					{from_eigen, Cdx},
					{from_eigen, residual_in_z_u},
					{from_eigen, residual_in_z_l},
					{from_eigen, d_dual_for_eq},
					{from_eigen, dual_for_eq},
					{from_eigen, d_primal_residual_eq},
					{from_eigen, primal_residual_eq},
					a_,
					n_in);

			// 3.4 if the argmin is within the interval [alpha[i],alpha[i+1]] is
			// stores the argmin and corresponding L2 norm

			if (i == 0) {
				if (a_ <= intervals[1]) {
					liste_norm_grad_interval.push_back(associated_grad_2_norm);
					liste_argmin.push_back(a_);
				}
			} else if (i == n_ - 2) {
				if (a_ >= intervals[usize(n_ - 2)]) {
					liste_norm_grad_interval.push_back(associated_grad_2_norm);
					liste_argmin.push_back(a_);
				}
			} else {
				if (a_ <= intervals[usize(i + 1)] && intervals[usize(i)] <= a_) {
					liste_norm_grad_interval.push_back(associated_grad_2_norm);
					liste_argmin.push_back(a_);
				}
			}
		}

		///////// STEP 4 ///////////
		// if the list of argmin obtained from intervals is not empty the
		// algorithm return the one minimizing the most the merit function
		// Otherwise, it returns the node minimizing the most the merit
		// function

		if (liste_norm_grad_interval.empty() == false) {

			std::vector<T> vec_norm_grad_interval{
					std::begin(liste_norm_grad_interval),
					std::end(liste_norm_grad_interval)};
			std::vector<T> vec_argmin{
					std::begin(liste_argmin), std::end(liste_argmin)};
			auto index =
					std::min_element(
							vec_norm_grad_interval.begin(), vec_norm_grad_interval.end()) -
					vec_norm_grad_interval.begin();

			alpha = vec_argmin[usize(index)];
		} else if (liste_norm_grad_noeud.empty() == false) {

			std::vector<T> vec_alphas{std::begin(alphas), std::end(alphas)};
			std::vector<T> vec_norm_grad_noeud{
					std::begin(liste_norm_grad_noeud), std::end(liste_norm_grad_noeud)};

			auto index = std::min_element(
											 vec_norm_grad_noeud.begin(), vec_norm_grad_noeud.end()) -
			             vec_norm_grad_noeud.begin();
			alpha = vec_alphas[usize(index)];
		}
	}

	return alpha;
}

template <typename T>
auto correction_guess_line_search_box(
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorView<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		qp::QpViewBox<T> qp) -> T {

	/*
	 * The function follows the algorithm designed by qpalm
	 * (see algorithm 2 : https://arxiv.org/pdf/1911.02934.pdf)
	 *
	 * To do so it does the following step
	 * 1/
	 * 1.1/ Store solutions of equations
	 * C(x+alpha dx) - l + ze/mu_in = 0
	 * C(x+alpha dx) - u + ze/mu_in = 0
	 *
	 * 1.2/ Prepare all needed algebra for gradient computation (and derive
	 * them only once) (TODO to add at a higher level in the solver)
	 * 1.3/ Sort the alpha
	 * 2/
	 * 2.1
	 * For each positive alpha compute the first derivative of
	 * phi(alpha) = [proximal augmented lagrangian of the subproblem evaluated
	 *               at x_k + alpha dx]
	 * using function "gradient_norm_qpalm_box"
	 * By construction for alpha = 0,
	 *   phi'(alpha) <= 0
	 *   and phi'(alpha) goes to infinity with alpha
	 * hence it cancels uniquely at one optimal alpha*
	 *
	 * while phi'(alpha)<=0 store the derivative (noted last_grad_neg) and
	 * alpha (last_alpha_neg)
	 * the first time phi'(alpha) > 0 store the derivative (noted
	 * first_grad_pos) and alpha (first_alpha_pos), and break the loo
	 *
	 * 2.2
	 * If first_alpha_pos corresponds to the first positive alpha of previous
	 * loop, then do
	 *   last_alpha_neg = 0
	 *   last_grad_neg = phi'(0)
	 * using function "gradient_norm_qpalm_box"
	 *
	 * 2.3
	 * the optimal alpha is within the interval
	 * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi' is
	 * an affine function in alph
	 * alpha* = alpha_last_neg
	 *        - last_neg_grad * (alpha_first_pos - alpha_last_neg) /
	 *                          (first_pos_grad - last_neg_grad);
	 */

	T machine_eps = std::numeric_limits<T>::epsilon();

	isize n_in = qp.C.rows;

	auto H = (qp.H).to_eigen();
	auto A = (qp.A).to_eigen();
	auto C = (qp.C).to_eigen();
	auto u = (qp.u).to_eigen();
	auto l = (qp.l).to_eigen();
	auto b = (qp.b).to_eigen();

	auto x_ = x.to_eigen();
	auto z_e = ze.to_eigen();
	auto y_e = ye.to_eigen();
	auto dx_ = dx.to_eigen();

	T alpha = 1;

	std::list<T> alphas = {};

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	Eigen::Matrix<T, Eigen::Dynamic, 1> Cdx = C * dx_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_in_z_u =
			(C * x_ - u + z_e / mu_in);
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_in_z_l =
			(C * x_ - l + z_e / mu_in);

	for (isize i = 0; i < n_in; i++) {
		if (Cdx(i) != 0) {
			alphas.push_back(-residual_in_z_u(i) / (Cdx(i) + machine_eps));
		}
		if (Cdx(i) != 0) {
			alphas.push_back(-residual_in_z_l(i) / (Cdx(i) + machine_eps));
		}
	}

	// 1.2 prepare all needed algebra for gradient norm computation (and derive
	// them only once) --> to add later in a workspace in qp_solve function

	Eigen::Matrix<T, Eigen::Dynamic, 1> Hdx = H * dx_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> Adx = A * dx_;
	Eigen::Matrix<T, Eigen::Dynamic, 1> residual_in_y = A * x_ - b + y_e / mu_eq;
	Eigen::Matrix<T, Eigen::Dynamic, 1> g = (qp.g).to_eigen();

	if (alphas.empty() == false) {
		// 1.3 sort the alphas
		alphas.sort();
		alphas.unique();

		////////// STEP 2 ///////////

		T last_neg_grad = 0;
		T alpha_last_neg = 0;
		T first_pos_grad = 0;
		T alpha_first_pos = 0;

		for (auto a : alphas) {

			if (a > 0) {
				if (a < T(1.e7)) {

					/*
					 * 2.1
					 * For each positive alpha compute the first derivative of
					 * phi(alpha) = [proximal augmented lagrangian of the
					 *               subproblem evaluated at x_k + alpha dx]
					 * using function "gradient_norm_qpalm_box"
					 *
					 * (By construction for alpha = 0,  phi'(alpha) <= 0 and
					 * phi'(alpha) goes to infinity with alpha hence it cancels
					 * uniquely at one optimal alpha*
					 *
					 * while phi'(alpha)<=0 store the derivative (noted
					 * last_grad_neg) and alpha (last_alpha_neg
					 * the first time phi'(alpha) > 0 store the derivative
					 * (noted first_grad_pos) and alpha (first_alpha_pos), and
					 * break the loop
					 */
					T gr = line_search::gradient_norm_qpalm_box(
							x,
							xe,
							dx,
							mu_eq,
							mu_in,
							rho,
							a,
							{from_eigen, Hdx},
							{from_eigen, g},
							{from_eigen, Adx},
							{from_eigen, residual_in_y},
							{from_eigen, residual_in_z_u},
							{from_eigen, residual_in_z_l},
							{from_eigen, Cdx},
							n_in);

					if (gr < 0) {
						alpha_last_neg = a;
						last_neg_grad = gr;
					} else {
						first_pos_grad = gr;
						alpha_first_pos = a;
						break;
					}
				}
			}
		}

		/*
		 * 2.2
		 * If first_alpha_pos corresponds to the first positive alpha of
		 * previous loop, then do
		 * last_alpha_neg = 0 and last_grad_neg = phi'(0) using function
		 * "gradient_norm_qpalm_box"
		 */
		if (last_neg_grad == T(0)) {
			alpha_last_neg = T(0);
			T gr = line_search::gradient_norm_qpalm_box(
					x,
					xe,
					dx,
					mu_eq,
					mu_in,
					rho,
					alpha_last_neg,
					{from_eigen, Hdx},
					{from_eigen, g},
					{from_eigen, Adx},
					{from_eigen, residual_in_y},
					{from_eigen, residual_in_z_u},
					{from_eigen, residual_in_z_l},
					{from_eigen, Cdx},
					n_in);
			last_neg_grad = gr;
		}

		/*
		 * 2.3
		 * the optimal alpha is within the interval
		 * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi'
		 * is an affine function in alpha
		 */
		alpha = alpha_last_neg - last_neg_grad *
		                             (alpha_first_pos - alpha_last_neg) /
		                             (first_pos_grad - last_neg_grad);
	}
	return alpha;
}

void active_set_change(
		VectorView<bool> new_active_set_,
		VectorViewMut<isize> current_bijection_map_,
		isize n_c,
		isize n_in) {

	/*
	 * arguments
	 * 1/ new_active_set : a vector which contains new active set of the
	 * problem, namely if
	 * new_active_set_u = Cx_k-u +z_k/mu_in>= 0
	 * new_active_set_l = Cx_k-l +z_k/mu_in<=
	 * then new_active_set = new_active_set_u OR new_active_set_
	 *
	 * 2/ current_bijection_map : a vector for which each entry corresponds to
	 * the current row of C of the current factorization
	 *
	 * for example, naming C_initial the initial C matrix of the problem, and
	 * C_current the one of the current factorization, the
	 * C_initial[i,:] = C_current[current_bijection_mal[i],:] for all
	 *
	 * 3/ n_c : the current number of active_inequalities
	 * This algorithm ensures that for all new version of C_current in the LDLT
	 * factorization all row index i < n_c correspond to current active indexes
	 * (all other correspond to inactive rows
	 *
	 * To do so,
	 * 1/ for initialization
	 * 1.1/ new_bijection_map = current_bijection_map
	 * 1.2/ n_c_f = n_
	 *
	 * 2/ All active indexes of the current bijection map (i.e
	 * current_bijection_map(i) < n_c by assumption) which are not active
	 * anymore in the new active set (new_active_set(i)=false are put at the
	 * end of new_bijection_map, i.
	 *
	 * 2.1/ for all j if new_bijection_map(j) > new_bijection_map(i), then
	 * new_bijection_map(j)-=1
	 * 2.2/ n_c_f -=1
	 * 2.3/ new_bijection_map(i) = n_in-1
	 *
	 * 3/ All active indexe of the new active set (new_active_set(i) == true)
	 * which are not active in the new_bijection_map (new_bijection_map(i) >=
	 * n_c_f) are put at the end of the current version of C, i.e
	 * 3.1/ if new_bijection_map(j) < new_bijection_map(i) &&
	 * new_bijection_map(j) >= n_c_f then new_bijection_map(j)+=1
	 * 3.2/ new_bijection_map(i) = n_c_f
	 * 3.3/ n_c_f +=1
	 *
	 * It returns finally the new_bijection_map, for which
	 * new_bijection_map(n_in) = n_c_f
	 */
	auto current_bijection_map = current_bijection_map_.as_const().to_eigen();
	auto new_active_set = new_active_set_.to_eigen();

	isize n_c_f = n_c;
	Eigen::Matrix<isize, Eigen::Dynamic, 1> new_bijection_map(n_in + 1);
	new_bijection_map.array().topRows(n_in) = current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set

	for (isize i = 0; i < n_in; i++) {
		if (current_bijection_map(i) < n_c) {
			if (new_active_set(i) == false) {
				for (isize j = 0; j < n_in; j++) {
					if (new_bijection_map(j) > new_bijection_map(i)) {
						new_bijection_map(j) -= 1;
					}
				}
				n_c_f -= 1;
				new_bijection_map(i) = n_in - 1;
			}
		}
	}

	// ajout au nouvel active set, suppression pour le nouvel unactive set

	for (isize i = 0; i < n_in; i++) {
		if (new_active_set(i) == true) {
			if (new_bijection_map(i) >= n_c_f) {

				for (isize j = 0; j < n_in; j++) {
					if (new_bijection_map(j) < new_bijection_map(i) &&
					    new_bijection_map(j) >= n_c_f) {
						new_bijection_map(j) += 1;
					}
				}
				new_bijection_map(i) = n_c_f;
				n_c_f += 1;
			}
		}
	}

	new_bijection_map(n_in) = n_c_f;
	current_bijection_map_.to_eigen() = new_bijection_map;
}
} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS */
