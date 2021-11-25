#ifndef INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS
#define INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS

#include <ldlt/ldlt.hpp>
#include "ldlt/views.hpp"
#include "qp/views.hpp"
#include <qp/QPWorkspace.hpp>
#include <qp/QPResults.hpp>
#include <qp/QPData.hpp>
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
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		MatrixView<T, LC> C,
		T alpha) -> T {

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
	qpwork._residual_in_z_u_plus_alpha = qpwork._primal_residual_in_scaled_u + alpha * qpwork._Cdx;
	qpwork._residual_in_z_l_plus_alpha = qpwork._primal_residual_in_scaled_l + alpha * qpwork._Cdx;
	qpwork._active_part_z = qpwork._ze + alpha * qpwork._dw_aug.tail(qpmodel._n_in);
	qpwork._rhs.head(qpmodel._dim) = qpwork._dual_residual_scaled + alpha * qpwork._d_dual_for_eq;
	qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq) = qpwork._primal_residual_eq_scaled + alpha * qpwork._d_primal_residual_eq;

	for (isize k = 0; k < qpmodel._n_in; ++k) {

		if (qpwork._residual_in_z_u_plus_alpha(k) >= 0) {
			if (qpwork._active_part_z(k) > 0) {
				qpwork._rhs.head(qpmodel._dim).noalias() += qpwork._active_part_z(k) * C_copy.row(k);
				qpwork._rhs(qpmodel._dim + qpmodel._n_eq + k) = qpwork._residual_in_z_u_plus_alpha(k) - qpwork._active_part_z(k) * qpresults._mu_in_inv; //mu stores the inverse of mu
			} else {
				qpwork._rhs(qpmodel._dim + qpmodel._n_eq + k) = qpwork._residual_in_z_u_plus_alpha(k);
			}

		} else if (qpwork._residual_in_z_l_plus_alpha(k) <= 0) {
			if (qpwork._active_part_z(k) < 0) {
				qpwork._rhs.head(qpmodel._dim).noalias() += qpwork._active_part_z(k) * C_copy.row(k);
				qpwork._rhs(qpmodel._dim + qpmodel._n_eq + k) = qpwork._residual_in_z_l_plus_alpha(k) - qpwork._active_part_z(k) * qpresults._mu_in_inv; //mu stores the inverse of mu
			} else {
				qpwork._rhs(qpmodel._dim + qpmodel._n_eq + k) = qpwork._residual_in_z_l_plus_alpha(k);
			}
		} else {
			qpwork._rhs(qpmodel._dim + qpmodel._n_eq + k) = qpwork._active_part_z(k);
		}
	}

	return qpwork._rhs.squaredNorm();
}

template <typename T>
auto gradient_norm_qpalm_box(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		T alpha,
		VectorView<T> g_) -> T {

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

	auto g = g_.to_eigen();
	// define active set
	qpwork._residual_in_z_u_plus_alpha = (qpwork._primal_residual_in_scaled_u + qpwork._Cdx * alpha);
	qpwork._residual_in_z_l_plus_alpha = (qpwork._primal_residual_in_scaled_l + qpwork._Cdx * alpha);

	qpwork._CTz = qpresults._rho * (qpresults._x - qpwork._xe) + g;

	T a(qpwork._dw_aug.head(qpmodel._dim).dot(qpwork._d_dual_for_eq) + ( qpwork._d_primal_residual_eq).squaredNorm() * qpresults._mu_eq + qpresults._rho * qpwork._dw_aug.head(qpmodel._dim).squaredNorm()); //mu stores mu
	T b(qpresults._x.dot(qpwork._d_dual_for_eq) + (qpwork._CTz).dot(qpwork._dw_aug.head(qpmodel._dim)) + ( qpwork._d_primal_residual_eq).dot(qpwork._d_primal_residual_eq) * qpresults._mu_eq) ; //mu stores mu

	for (isize k = 0; k < qpmodel._n_in; ++k) {
		
		if (qpwork._residual_in_z_u_plus_alpha(k) > 0) {

			a += qp::detail::square(qpwork._Cdx(k)) * qpresults._mu_in ; //mu stores  mu 
			b += qpwork._Cdx(k) * qpwork._primal_residual_in_scaled_u(k) * qpresults._mu_in; //mu stores  mu

		} 
		else if (qpwork._residual_in_z_l_plus_alpha(k) < 0) {

			a += qp::detail::square(qpwork._Cdx(k)) * qpresults._mu_in; //mu stores  mu
			b += qpwork._Cdx(k) * qpwork._primal_residual_in_scaled_l(k) * qpresults._mu_in; //mu stores mu
		}
	}

	return a * alpha + b;
}

template <typename T, Layout LC>
auto local_saddle_point_box(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		MatrixView<T, LC> C,
		T& alpha) -> T {
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

	qpwork._residual_in_z_u_plus_alpha = (qpwork._primal_residual_in_scaled_u + alpha * qpwork._Cdx);
	qpwork._residual_in_z_l_plus_alpha = (qpwork._primal_residual_in_scaled_l + alpha * qpwork._Cdx);
	qpwork._active_part_z = (qpwork._ze + alpha * qpwork._dw_aug.tail(qpmodel._n_in));
	
	// a0 computation

	T a0(qpwork._d_primal_residual_eq.squaredNorm());
	T b0(qpwork._primal_residual_eq_scaled.dot(qpwork._d_primal_residual_eq));
	T c0(qpwork._primal_residual_eq_scaled.squaredNorm());

	for (isize k = 0; k < qpmodel._n_in; ++k) {

		if (qpwork._residual_in_z_u_plus_alpha(k) >= 0) {

			if (qpwork._active_part_z(k) > 0) {

				qpwork._d_dual_for_eq += qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * C_copy.row(k);
				qpwork._dual_residual_scaled += qpwork._ze(k) * C_copy.row(k);
				a0 += qp::detail::square(qpwork._Cdx(k) - qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * qpresults._mu_in_inv); //mu stores the inverse of mu
				b0 += (qpwork._Cdx(k) - qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * qpresults._mu_in_inv) * (qpwork._primal_residual_in_scaled_u(k) - qpwork._ze(k) * qpresults._mu_in_inv); //mu stores the inverse of mu
				c0 += qp::detail::square(qpwork._primal_residual_in_scaled_u(k) - qpwork._ze(k) * qpresults._mu_in_inv); //mu stores the inverse of mu

			} else {

				a0 += qp::detail::square(qpwork._Cdx(k));
				b0 += qpwork._Cdx(k) * qpwork._primal_residual_in_scaled_u(k);
				c0 += qp::detail::square(qpwork._primal_residual_in_scaled_u(k));
			}

		} else if (qpwork._residual_in_z_l_plus_alpha(k) <= 0) {

			if (qpwork._active_part_z(k) < 0) {

				qpwork._d_dual_for_eq += qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * C_copy.row(k);
				qpwork._dual_residual_scaled += qpwork._ze(k) * C_copy.row(k);
				a0 += qp::detail::square(qpwork._Cdx(k) - qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * qpresults._mu_in_inv); //mu stores the inverse of mu
				b0 += (qpwork._Cdx(k) - qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * qpresults._mu_in_inv) * (qpwork._primal_residual_in_scaled_l(k) - qpwork._ze(k) * qpresults._mu_in_inv); //mu stores the inverse of mu
				c0 += qp::detail::square(qpwork._primal_residual_in_scaled_l(k) - qpwork._ze(k) * qpresults._mu_in_inv); //mu stores the inverse of mu

			} else {

				a0 += qp::detail::square(qpwork._Cdx(k));
				b0 += qpwork._Cdx(k) * qpwork._primal_residual_in_scaled_l(k);
				c0 += qp::detail::square(qpwork._primal_residual_in_scaled_l(k));
			}

		} else {
			a0 += qp::detail::square(qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k));
			b0 += qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+k) * qpwork._ze(k);
			c0 += qp::detail::square(qpwork._ze(k));
		}
	}
	a0 += qpwork._d_dual_for_eq.squaredNorm();
	c0 += qpwork._dual_residual_scaled.squaredNorm();
	b0 += qpwork._d_dual_for_eq.dot(qpwork._dual_residual_scaled);
	b0 *= 2;

	// derivation of the loss function value and corresponding argmin alpha

	T res = 0;

	if (a0 != 0) {
		alpha = (-b0 / (2 * a0));
		res = a0 * qp::detail::square(alpha) + b0 * alpha + c0;
	} else if (b0 != 0) {
		alpha = (-c0 / (b0));
		res = b0 * alpha + c0;
	} else {
		alpha = 0;
		res = c0;
	}

	return res;
}

template <typename T,Layout LC>
auto initial_guess_LS(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		MatrixView<T,LC> C,
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

	static constexpr T machine_eps = std::numeric_limits<T>::epsilon();
	static constexpr T machine_inf = std::numeric_limits<T>::infinity();

	T alpha = 1;
	T alpha_n = 1;
	T gr_n = 1 ;
	T alpha_interval = 1;
	T gr_interval = 1;
	T alpha_(0);
	isize n_alpha(0);

	/////////// STEP 1 ////////////
	// computing the "nodes" alphas which cancel  C.dot(xe+alpha dx) - u,
	// C.dot(xe+alpha dx) - l and ze + alpha dz  /////////////

	qpwork.alphas.clear();

	//std::list<T> alphas = {}; // TODO use a vector instead of a list
	// 1.1 add solutions of equation z+alpha dz = 0

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (std::abs(qpwork._ze(i)) != 0) {

			alpha_ = -qpwork._ze(i) / (qpwork._dw_aug(qpmodel._dim+qpmodel._n_eq+i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				qpwork.alphas.push_back(alpha_);
			}
		}
	}

	// 1.1 add solutions of equations C(x+alpha dx)-u +ze/mu_in = 0 and C(x+alpha
	// dx)-l +ze/mu_in = 0

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (std::abs(qpwork._Cdx(i)) != 0) {
			alpha_ = -qpwork._primal_residual_in_scaled_u(i) / (qpwork._Cdx(i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				qpwork.alphas.push_back(alpha_);
			}
			alpha_ = -qpwork._primal_residual_in_scaled_l(i) / (qpwork._Cdx(i) + machine_eps);
			if (std::abs(alpha_) < ball_radius) {
				qpwork.alphas.push_back(alpha_);
			}
		}
	}

	n_alpha = qpwork.alphas.size();
	// 1.2 it prepares all needed algebra in order not to derive it each time

	if (!qpwork.alphas.empty()) {
		//////// STEP 2 ////////
		// 2.1/ it sorts alpha nodes
		
		std::sort (qpwork.alphas.begin(), qpwork.alphas.begin()+n_alpha); 

		// 2.2/ for each node active set and associated gradient are computed

		bool first = true;
		for (isize i = 0; i < n_alpha; ++i) {
			alpha_ = qpwork.alphas[i];
			if (std::abs(alpha_) < ball_radius) {
				
				
				// calcul de la norm du gradient du noeud

				T grad_norm = line_search::gradient_norm_computation_box(
						qpwork,
						qpresults,
						qpmodel,
						C,
						alpha_);

				if (first){
					gr_n = grad_norm;
					alpha_n = alpha_;
					first = false;
				}else{
					if (grad_norm< gr_n){
						gr_n = grad_norm;
						alpha_n = alpha_ ;
					}
				}
			} 
		}

		//////////STEP 3 ////////////
		// 3.1 : define intervals with alphas
		
		first = true;

		for (isize i = 0; i < n_alpha + 1; ++i) {

			// 3.2 : it derives the mean node (alpha[i]+alpha[i+1])/2
			// the corresponding active sets active_inequalities_u and
			// active_inequalities_l cap ze and dz is derived through function
			// local_saddle_point_box
			if (i == 0){
				alpha_ = (2*qpwork.alphas[0]-1) / 2;
			} else if (i==n_alpha){
				alpha_ = (  2*qpwork.alphas[n_alpha-1] +1 ) /2;
			} else{
				alpha_ = (qpwork.alphas[i] + qpwork.alphas[i + 1]) / 2;
			}

			// 3.3 on this interval the merit function is a second order
			// polynomial in alpha
			// the function "local_saddle_point_box" derives the exact minimum
			// and corresponding merit function L2 norm (for this minimum)

			T associated_grad_2_norm = line_search::local_saddle_point_box(
					qpwork,
					qpresults,
					qpmodel,
					C,
					alpha_);

			// 3.4 if the argmin is within the interval [alpha[i],alpha[i+1]] is
			// stores the argmin and corresponding L2 norm

			if (i == 0) {
				if (alpha_ <= qpwork.alphas[0]) {
					if (first){
						first = false;
						alpha_interval = alpha_ ;
						gr_interval = associated_grad_2_norm;
					} else{
						if (associated_grad_2_norm<gr_interval){
							alpha_interval = alpha_ ;
							gr_interval = associated_grad_2_norm;
						}
					}
				}
			} else if (i == n_alpha) {
				if (alpha_ >= qpwork.alphas[n_alpha - 1]) {
					if (first){
						first = false;
						alpha_interval = alpha_ ;
						gr_interval = associated_grad_2_norm;
					} else{
						if (associated_grad_2_norm<gr_interval){
							alpha_interval = alpha_ ;
							gr_interval = associated_grad_2_norm;
						}
					}
				}
			} else {
				if (alpha_ <= qpwork.alphas[i + 1] && qpwork.alphas[i] <= alpha_) {
					if (first){
						first = false;
						alpha_interval = alpha_ ;
						gr_interval = associated_grad_2_norm;
					} else{
						if (associated_grad_2_norm<gr_interval){
							alpha_interval = alpha_ ;
							gr_interval = associated_grad_2_norm;
						}
					}
				}
			}
		}
		///////// STEP 4 ///////////
		// if the list of argmin obtained from intervals is not empty the
		// algorithm return the one minimizing the most the merit function
		// Otherwise, it returns the node minimizing the most the merit
		// function

		if (gr_interval!= 1){
			alpha = alpha_interval;
		}else if (gr_n!=1){
			alpha = alpha_n;
		}
	}

	return alpha;
}

template <typename T>
auto correction_guess_LS(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		VectorView<T> g) -> T {

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

	static constexpr T machine_eps = std::numeric_limits<T>::epsilon();

	T alpha = 1;

	qpwork.alphas.clear();
	isize n_alpha(0);
	T alpha_(0);

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (qpwork._Cdx(i) != 0) {
			qpwork.alphas.push_back(-qpwork._primal_residual_in_scaled_u(i) / (qpwork._Cdx(i) + machine_eps));
			qpwork.alphas.push_back(-qpwork._primal_residual_in_scaled_l(i) / (qpwork._Cdx(i) + machine_eps));
		}
	}
	n_alpha = qpwork.alphas.size();
	if (!qpwork.alphas.empty()) {
		// 1.2 sort the alphas
		std::sort (qpwork.alphas.begin(), qpwork.alphas.begin()+n_alpha); 

		////////// STEP 2 ///////////

		T last_neg_grad = 0;
		T alpha_last_neg = 0;
		T first_pos_grad = 0;
		T alpha_first_pos = 0;

		for (isize i = 0; i < n_alpha; ++i) {

			alpha_ = qpwork.alphas[i];
			if (alpha_ > machine_eps) {
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
							qpwork,
							qpresults,
							qpmodel,
							alpha_,
							g);

					if (gr < 0) {
						alpha_last_neg = alpha_;
						last_neg_grad = gr;
					} else {
						first_pos_grad = gr;
						alpha_first_pos = alpha_;
						break;
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
		if (last_neg_grad == 0) {
			alpha_last_neg = 0;
			T gr = line_search::gradient_norm_qpalm_box(
					qpwork,
					qpresults,
					qpmodel,
					alpha_last_neg,
					g);
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

template <typename T>
void active_set_change_new(
		qp::Qpworkspace<T>& qpwork,
		qp::Qpresults<T>& qpresults,
		qp::Qpdata<T>& qpmodel,
		qp::QpViewBox<T> qp) {

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
			
	//auto current_bijection_map = current_bijection_map_.as_const().to_eigen();
	//auto new_active_set = new_active_set_.to_eigen();
	//auto dw = dw_.to_eigen();
	

	isize n_c_f = qpresults._n_c;
	//Eigen::Matrix<isize, Eigen::Dynamic, 1> new_bijection_map(n_in);
	qpwork._new_bijection_map = qpwork._current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (qpwork._current_bijection_map(i) < qpresults._n_c) {
			if (!qpwork._active_inequalities(i)) {
				// delete current_bijection_map(i)
				qpwork._ldl.delete_at(qpwork._new_bijection_map(i) + qpmodel._dim + qpmodel._n_eq);
				for (isize j = 0; j < qpmodel._n_in; j++) {
					if (qpwork._new_bijection_map(j) > qpwork._new_bijection_map(i)) {
						qpwork._new_bijection_map(j) -= 1;
					}
				}
				n_c_f -= 1;
				qpwork._new_bijection_map(i) = qpmodel._n_in - 1;
			}
		}
	}

	// ajout au nouvel active set, suppression pour le nouvel unactive set

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (qpwork._active_inequalities(i)) {
			if (qpwork._new_bijection_map(i) >= n_c_f) {
				// add at the end
				
				auto C_ = qp.C.to_eigen();
				qpwork._dw_aug.segment(qpmodel._dim,n_c_f+qpmodel._n_eq).setZero();
				qpwork._dw_aug.head(qpmodel._dim) = C_.row(i);
				qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + n_c_f) = - qpresults._mu_in_inv; // mu stores the inverse of mu
				qpwork._ldl.insert_at(qpmodel._n_eq + qpmodel._dim + n_c_f, qpwork._dw_aug.head(n_c_f+1+qpmodel._n_eq+qpmodel._dim));

				for (isize j = 0; j < qpmodel._n_in; j++) {
					if (qpwork._new_bijection_map(j) < qpwork._new_bijection_map(i) &&
						qpwork._new_bijection_map(j) >= n_c_f) {
						qpwork._new_bijection_map(j) += 1;
					}
				}
				qpwork._new_bijection_map(i) = n_c_f;
				n_c_f += 1;
				

			}
		}
	}
	qpresults._n_c = n_c_f;
	qpwork._current_bijection_map = qpwork._new_bijection_map;
}

} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_LINE_SEARCH_HPP_2TUXO5DFS */