#ifndef INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS
#define INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS

#include <ldlt/ldlt.hpp>
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

template <typename T, Layout LC>
auto gradient_norm_computation_box(
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

	auto dz = dz_.to_eigen();
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
	LDLT_MULTI_WORKSPACE_MEMORY(
			(tmp_u_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(tmp_l_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(active_part_z_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, T),
			(res_, Init, Vec(dim + n_eq + n_in), LDLT_CACHELINE_BYTES, T));

	auto tmp_u = tmp_u_.to_eigen();
	auto tmp_l = tmp_l_.to_eigen();
	auto active_part_z = active_part_z_.to_eigen();
	auto res = res_.to_eigen();

	tmp_u = residual_in_z_u + alpha * Cdx;
	tmp_l = residual_in_z_l + alpha * Cdx;

	active_part_z = z_e + alpha * dz;

	res.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq;

	res.middleRows(dim, n_eq) = primal_residual_eq + alpha * d_primal_residual_eq;

	for (isize k = 0; k < n_in; ++k) {

		if (tmp_u(k) >= T(0.)) {
			if (active_part_z(k) > T(0.)) {
				res.topRows(dim) += active_part_z(k) * C_copy.row(k);
				res(dim + n_eq + k) = tmp_u(k) - active_part_z(k) / mu_in;
			} else {
				res(dim + n_eq + k) = tmp_u(k);
			}

		} else if (tmp_l(k) <= T(0.)) {
			if (active_part_z(k) < T(0.)) {
				res.topRows(dim) += active_part_z(k) * C_copy.row(k);
				res(dim + n_eq + k) = tmp_l(k) - active_part_z(k) / mu_in;
			} else {
				res(dim + n_eq + k) = tmp_l(k);
			}
		} else {
			res(dim + n_eq + k) = active_part_z(k);
		}
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

	T a(dx_.dot(Hdx) + mu_eq * (Adx).squaredNorm() + rho * dx_.squaredNorm());
	T b(x_.dot(Hdx) + (rho * (x_ - xe_) + g).dot(dx_) +
	    mu_eq * (Adx).dot(residual_in_y));

	for (isize k = 0; k < n_in; ++k) {

		if (tmp_u(k) > T(0.)) {

			a += mu_in * pow(Cdx(k), T(2));
			b += mu_in * Cdx(k) * residual_in_z_u(k);

		} else if (tmp_l(k) < T(0.)) {

			a += mu_in * pow(Cdx(k), T(2));
			b += mu_in * Cdx(k) * residual_in_z_l(k);
		}
	}

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
		T& alpha,
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
	auto active_part = z_e + alpha * dz;

	// a0 computation

	T a0(d_primal_residual_eq.squaredNorm());
	T b0(primal_residual_eq.dot(d_primal_residual_eq));
	T c0(primal_residual_eq.squaredNorm());

	for (isize k = 0; k < n_in; ++k) {

		if (tmp_u(k) >= T(0.)) {

			if (active_part(k) > T(0)) {

				d_dual_for_eq += dz(k) * C_copy.row(k);
				dual_for_eq += z_e(k) * C_copy.row(k);
				a0 += pow(Cdx(k) - dz(k) / mu_in, T(2));
				b0 += (Cdx(k) - dz(k) / mu_in) * (residual_in_z_u(k) - z_e(k) / mu_in);
				c0 += pow(residual_in_z_u(k) - z_e(k) / mu_in, T(2));

			} else {

				a0 += pow(Cdx(k), T(2));
				b0 += Cdx(k) * residual_in_z_u(k);
				c0 += pow(residual_in_z_u(k), T(2));
			}

		} else if (tmp_l(k) <= T(0.)) {

			if (active_part(k) < T(0)) {

				d_dual_for_eq += dz(k) * C_copy.row(k);
				dual_for_eq += z_e(k) * C_copy.row(k);
				a0 += pow(Cdx(k) - dz(k) / mu_in, T(2));
				b0 += (Cdx(k) - dz(k) / mu_in) * (residual_in_z_l(k) - z_e(k) / mu_in);
				c0 += pow(residual_in_z_l(k) - z_e(k) / mu_in, T(2));

			} else {

				a0 += pow(Cdx(k), T(2));
				b0 += Cdx(k) * residual_in_z_l(k);
				c0 += pow(residual_in_z_l(k), T(2));
			}

		} else {
			a0 += pow(dz(k), T(2));
			b0 += dz(k) * z_e(k);
			c0 += pow(z_e(k), T(2));
		}
	}
	a0 += d_dual_for_eq.squaredNorm();
	c0 += dual_for_eq.squaredNorm();
	b0 += d_dual_for_eq.dot(dual_for_eq);
	b0 *= T(2);

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

template <typename T, Layout LC>
auto initial_guess_LS(
		VectorView<T> ze,
		VectorView<T> dz,
		VectorView<T> residual_in_z_l_,
		VectorView<T> residual_in_z_u_,
		VectorView<T> Cdx_,
		VectorView<T> d_dual_for_eq,
		VectorView<T> dual_for_eq,
		VectorView<T> d_primal_residual_eq,
		VectorView<T> primal_residual_eq,
		MatrixView<T, LC> C,
		T mu_eq,
		T mu_in,
		T rho,
		isize dim,
		isize n_eq,
		isize n_in,
		T ball_radius) -> T {
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
	 * algorithm return the one minimizing the most the merit function
	 * Otherwise, it returns the node minimizing the most the merit function
	 */

	T machine_eps = std::numeric_limits<T>::epsilon();
	T machine_inf = std::numeric_limits<T>::infinity();

	auto z_e = ze.to_eigen();
	auto dz_ = dz.to_eigen();

	T alpha = 1;

	T alpha_(0);
	/////////// STEP 1 ////////////
	// computing the "nodes" alphas which cancel  C.dot(xe+alpha dx) - u,
	// C.dot(xe+alpha dx) - l and ze + alpha dz  /////////////

	std::list<T> alphas = {}; // TODO use a vector instead of a list
	// 1.1 add solutions of equation z+alpha dz = 0

	for (isize i = 0; i < n_in; i++) {
		if (std::abs(z_e(i)) != 0) {
			alpha_ = -z_e(i) / (dz_(i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				alphas.push_back(alpha_);
			}
			// alphas.push_back(-z_e(i) / (dz_(i) + machine_eps));
		}
	}

	// 1.1 add solutions of equations C(x+alpha dx)-u +ze/mu_in = 0 and C(x+alpha
	// dx)-l +ze/mu_in = 0

	auto Cdx = Cdx_.to_eigen();
	auto residual_in_z_u = residual_in_z_u_.to_eigen();
	auto residual_in_z_l = residual_in_z_l_.to_eigen();

	for (isize i = 0; i < n_in; i++) {
		if (std::abs(Cdx(i)) != 0) {
			alpha_ = -residual_in_z_u(i) / (Cdx(i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				alphas.push_back(alpha_);
			}
			alpha_ = -residual_in_z_l(i) / (Cdx(i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				alphas.push_back(alpha_);
			}
			// alphas.push_back(-residual_in_z_u(i) / (Cdx(i) + machine_eps));
			// alphas.push_back(-residual_in_z_l(i) / (Cdx(i) + machine_eps));
		}
	}

	// 1.2 it prepares all needed algebra in order not to derive it each time

	if (!alphas.empty()) {
		//////// STEP 2 ////////
		// 2.1/ it sorts alpha nodes
		alphas.sort();
		alphas.unique();

		// 2.2/ for each node active set and associated gradient are computed

		std::list<T> liste_norm_grad_noeud = {};
		/*
		std::cout << "residual_in_z_l_ " << residual_in_z_l  << std::endl;
		std::cout << "residual_in_z_u_ " << residual_in_z_u  << std::endl;
		std::cout << "Cdx " << Cdx << std::endl;
		std::cout << "d_dual_for_eq " << d_dual_for_eq.to_eigen() << std::endl;
		std::cout << "dual_for_eq " << dual_for_eq.to_eigen()  << std::endl;
		std::cout << "d_primal_residual_eq " << d_primal_residual_eq.to_eigen()  <<
		std::endl; std::cout << "primal_residual_eq " <<
		primal_residual_eq.to_eigen()  << std::endl; std::cout << "dz_ " << dz_ <<
		std::endl;
		*/
		for (auto a : alphas) {

			if (std::abs(a) < ball_radius) {

				// calcul de la norm du gradient du noeud
				T grad_norm = line_search::gradient_norm_computation_box(
						ze,
						dz,
						mu_in,
						C,
						Cdx_,
						residual_in_z_u_,
						residual_in_z_l_,
						d_dual_for_eq,
						dual_for_eq,
						d_primal_residual_eq,
						primal_residual_eq,
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
					dz,
					mu_in,
					C,
					Cdx_,
					residual_in_z_u_,
					residual_in_z_l_,
					d_dual_for_eq,
					dual_for_eq,
					d_primal_residual_eq,
					primal_residual_eq,
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

		if (!liste_norm_grad_interval.empty()) {

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
		} else if (!liste_norm_grad_noeud.empty()) {

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
auto correction_guess_LS(
		VectorViewMut<T> Hdx,
		VectorViewMut<T> Adx,
		VectorViewMut<T> Cdx,
		VectorViewMut<T> residual_in_y,
		VectorViewMut<T> residual_in_z_u,
		VectorViewMut<T> residual_in_z_l,
		VectorView<T> dx,
		VectorView<T> g,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		T mu_eq,
		T mu_in,
		T rho,
		isize n_in) -> T {

	auto Hdx_ = Hdx.to_eigen();
	auto Adx_ = Adx.to_eigen();
	auto Cdx_ = Cdx.to_eigen();
	auto residual_in_y_ = residual_in_y.to_eigen();
	auto residual_in_z_u_ = residual_in_z_u.to_eigen();
	auto residual_in_z_l_ = residual_in_z_l.to_eigen();

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
	 * 1.2/ Sort the alpha
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

	auto x_ = x.to_eigen();
	auto z_e = ze.to_eigen();
	auto y_e = ye.to_eigen();

	T alpha = 1;

	std::list<T> alphas = {};

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	for (isize i = 0; i < n_in; i++) {
		if (Cdx_(i) != 0) {
			alphas.push_back(-residual_in_z_u_(i) / (Cdx_(i) + machine_eps));
		}
		if (Cdx_(i) != 0) {
			alphas.push_back(-residual_in_z_l_(i) / (Cdx_(i) + machine_eps));
		}
	}

	if (!alphas.empty()) {
		// 1.2 sort the alphas
		alphas.sort();
		alphas.unique();

		////////// STEP 2 ///////////

		T last_neg_grad = 0;
		T alpha_last_neg = 0;
		T first_pos_grad = 0;
		T alpha_first_pos = 0;

		for (auto a : alphas) {

			if (a > machine_eps) {
				if (a < T(1.e21)) {

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
							// VectorView<T>{from_eigen,dx},
							dx,
							mu_eq,
							mu_in,
							rho,
							a,
							VectorView<T>{from_eigen, Hdx_},
							g,
							VectorView<T>{from_eigen, Adx_},
							VectorView<T>{from_eigen, residual_in_y_},
							VectorView<T>{from_eigen, residual_in_z_u_},
							VectorView<T>{from_eigen, residual_in_z_l_},
							VectorView<T>{from_eigen, Cdx_},
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
					// VectorView<T>{from_eigen,dx},
					dx,
					mu_eq,
					mu_in,
					rho,
					alpha_last_neg,
					VectorView<T>{from_eigen, Hdx_},
					g,
					VectorView<T>{from_eigen, Adx_},
					VectorView<T>{from_eigen, residual_in_y_},
					VectorView<T>{from_eigen, residual_in_z_u_},
					VectorView<T>{from_eigen, residual_in_z_l_},
					VectorView<T>{from_eigen, Cdx_},
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
		// std::cout << "alpha_last_neg " << alpha_last_neg << " alpha_first_pos "
		// << alpha_first_pos << " last_neg_grad " << last_neg_grad << "
		// first_pos_grad " <<first_pos_grad<< std::endl;
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
	new_bijection_map.topRows(n_in) = current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set

	for (isize i = 0; i < n_in; i++) {
		if (current_bijection_map(i) < n_c) {
			if (!new_active_set(i)) {

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
		if (new_active_set(i)) {
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

template <typename T>
void active_set_change_new(
		VectorView<bool> new_active_set_,
		VectorViewMut<isize> current_bijection_map_,
		isize& n_c,
		isize n_in,
		isize dim,
		isize n_eq,
		ldlt::Ldlt<T>& ldl,
		qp::QpViewBox<T> qp,
		T mu_in,
		T mu_eq,
		T rho,
		isize& deletion,
		isize& adding) {

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
	Eigen::Matrix<isize, Eigen::Dynamic, 1> new_bijection_map(n_in);
	new_bijection_map = current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set
	//deletion = isize(0);
	for (isize i = 0; i < n_in; i++) {
		if (current_bijection_map(i) < n_c) {
			if (!new_active_set(i)) {
				// delete current_bijection_map(i)
				ldl.delete_at(new_bijection_map(i) + dim + n_eq);
				deletion+=1;
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
	//adding = isize(0);
	for (isize i = 0; i < n_in; i++) {
		if (new_active_set(i)) {
			if (new_bijection_map(i) >= n_c_f) {
				// add at the end
				[&] {
					LDLT_MULTI_WORKSPACE_MEMORY(
							(row_,
					     Init,
					     Vec(n_c_f + 1 + n_eq + dim),
					     LDLT_CACHELINE_BYTES,
					     T));
					auto row = row_.to_eigen();
					auto C_ = qp.C.to_eigen();
					row.topRows(dim) = C_.row(i);
					row(dim + n_eq + n_c_f) = -T(1) / mu_in;
					ldl.insert_at(n_eq + dim + n_c_f, row);
					adding+=1;
					for (isize j = 0; j < n_in; j++) {
						if (new_bijection_map(j) < new_bijection_map(i) &&
						    new_bijection_map(j) >= n_c_f) {
							new_bijection_map(j) += 1;
						}
					}
					new_bijection_map(i) = n_c_f;
					n_c_f += 1;
				}();

			}
		}
	}
	n_c = n_c_f;
	current_bijection_map_.to_eigen() = new_bijection_map;
}

} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS */