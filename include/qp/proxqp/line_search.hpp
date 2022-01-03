#ifndef INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS
#define INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS

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
auto gradient_norm_computation_box(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork,
		T alpha
		) -> T {

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

	qpwork._primal_residual_in_scaled_up_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_up + alpha * qpwork._Cdx;
	qpwork._primal_residual_in_scaled_low_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_low + alpha * qpwork._Cdx;
	qpwork._active_part_z.noalias() = qpwork._z_prev + alpha * qpwork._dw_aug.tail(qpmodel._n_in);
	qpwork._rhs.head(qpmodel._dim).noalias() = qpwork._dual_residual_scaled + alpha * qpwork._Hdx;
	qpwork._rhs.segment(qpmodel._dim, qpmodel._n_eq).noalias() = qpwork._primal_residual_eq_scaled + alpha * qpwork._Adx;
	
	// stores [qpwork._active_part_z]_act
	qpwork._err.tail(qpmodel._n_in).noalias() = ((qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.) && qpwork._active_part_z.array()>T(0.)) || (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() <= T(0.) && qpwork._active_part_z.array()<T(0.)) ).select(qpwork._active_part_z, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	// use it to derive CT [qpwork._active_part_z]_act
	qpwork._rhs.head(qpmodel._dim).noalias() += qpwork._c_scaled.transpose() * qpwork._err.tail(qpmodel._n_in) ;
	// define [qpwork._active_part_z]_act / mu_in
	qpwork._err.tail(qpmodel._n_in) *= qpresults._mu_in_inv;

	qpwork._rhs.tail(qpmodel._n_in).noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.)).select(qpwork._primal_residual_in_scaled_up , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in))
							+ (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() <= T(0.)).select(qpwork._primal_residual_in_scaled_low , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in))
							- (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() < T(0.) && qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() > T(0.)).select(qpwork._active_part_z , qpwork._err.tail(qpmodel._n_in)) ;


	return qpwork._rhs.squaredNorm();
}

template <typename T>
auto gradient_norm_qpalm_box(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork,
		T alpha
		) -> T {

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

	qpwork._primal_residual_in_scaled_up_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_up + qpwork._Cdx * alpha;
	qpwork._primal_residual_in_scaled_low_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_low + qpwork._Cdx * alpha;

	T a( qpwork._dw_aug.head(qpmodel._dim).dot(qpwork._Hdx) + qpresults._mu_eq * (qpwork._Adx).squaredNorm()+qpresults._rho *  qpwork._dw_aug.head(qpmodel._dim).squaredNorm()    );
	qpwork._err.head(qpmodel._dim).noalias() = qpresults._rho * ( qpresults._x - qpwork._x_prev) + qpwork._g_scaled;
	T b( qpresults._x.dot(qpwork._Hdx) + (qpwork._err.head(qpmodel._dim)).dot( qpwork._dw_aug.head(qpmodel._dim)) +
	     qpresults._mu_eq * (qpwork._Adx).dot(qpwork._primal_residual_eq_scaled) ) ; 

	// derive Cdx_act
	qpwork._err.tail(qpmodel._n_in).noalias() = ((qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.)) || (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.)) ).select(qpwork._Cdx, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));

	a+= qpresults._mu_in * qpwork._err.tail(qpmodel._n_in).squaredNorm();

	// derive vector [Cx-u+ze/mu]_+ + [Cx-l+ze/mu]--
	qpwork._active_part_z.noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.)).select(qpwork._primal_residual_in_scaled_up , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in))
							+ (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.)).select(qpwork._primal_residual_in_scaled_low , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in)) ; 

	b+=  qpresults._mu_in * qpwork._active_part_z.dot(qpwork._err.tail(qpmodel._n_in));
	

	return a * alpha + b;
}

template <typename T>
auto local_saddle_point_box(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork,
		T& alpha
		) -> T {
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

	qpwork._primal_residual_in_scaled_up_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_up + alpha * qpwork._Cdx;
	qpwork._primal_residual_in_scaled_low_plus_alphaCdx.noalias() = qpwork._primal_residual_in_scaled_low + alpha * qpwork._Cdx;
	qpwork._active_part_z.noalias() = (qpwork._z_prev + alpha * qpwork._dw_aug.tail(qpmodel._n_in));
	
	// a0 computation

	T a0(qpwork._Adx.squaredNorm());
	T b0(qpwork._primal_residual_eq_scaled.dot(qpwork._Adx));
	T c0(qpwork._primal_residual_eq_scaled.squaredNorm());

	qpwork._rhs.head(qpmodel._dim) = qpwork._Hdx;
	qpwork._CTz = qpwork._dual_residual_scaled;

	///// derive ||[Cdx-dz/mu_in]_act||**2 with act = C(x+alpha*dx)-u+ze/mu > 0 or C(x+alpha*dx)-ze/mu <0 and with dz_act = 0 iff [z+alpha*dz]_Iu <0 or [z+alpha*dz]_Il >0
	
	// derive dz_act
	qpwork._rhs.tail(qpmodel._n_in).noalias() = ((qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.) && qpwork._active_part_z.array()>T(0.)) || (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() <= T(0.) && qpwork._active_part_z.array()<T(0.)) ).select(qpwork._dw_aug.tail(qpmodel._n_in), Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	/// use it to compute CT*dz_act
	qpwork._rhs.head(qpmodel._dim).noalias() += qpwork._c_scaled.transpose() * qpwork._rhs.tail(qpmodel._n_in);
	// derive Cdx_act
	qpwork._err.tail(qpmodel._n_in).noalias() = ((qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.)) || (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() <= T(0.)) ).select(qpwork._Cdx, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	// derive [Cdx-dz/mu_in]_act
	qpwork._err.tail(qpmodel._n_in).noalias() -= qpwork._rhs.tail(qpmodel._n_in) * qpresults._mu_in_inv;

	a0 += qpwork._rhs.head(qpmodel._dim).squaredNorm();
	a0 += qpwork._err.tail(qpmodel._n_in).squaredNorm();
	// dz_inact
	qpwork._rhs.tail(qpmodel._n_in).noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() < T(0.) || qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() > T(0.) ).select(qpwork._dw_aug.tail(qpmodel._n_in), Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	a0 += qpwork._rhs.tail(qpmodel._n_in).squaredNorm();

	// derive z_act
	qpwork._rhs.tail(qpmodel._n_in).noalias() = ((qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.) && qpwork._active_part_z.array()>T(0.)) || (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() <= T(0.) && qpwork._active_part_z.array()<T(0.)) ).select(qpwork._z_prev, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	/// use it to compute CT*z_act
	qpwork._CTz.noalias() += qpwork._c_scaled.transpose() * qpwork._rhs.tail(qpmodel._n_in);
	c0 += qpwork._CTz.squaredNorm();
	b0 += qpwork._rhs.head(qpmodel._dim).dot(qpwork._CTz); // d_dual + CT*dz_act dot dual + CT*z_act
	// derive vector [Cx-u+(ze-z_act)/mu]_+ + [Cx-l+(ze-z_act)/mu]--
	qpwork._rhs.tail(qpmodel._n_in) *= qpresults._mu_in_inv;
	qpwork._active_part_z.noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() >= T(0.)).select(qpwork._primal_residual_in_scaled_up , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in))
							+ (qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() <= T(0.)).select(qpwork._primal_residual_in_scaled_low , Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in))
							- qpwork._rhs.tail(qpmodel._n_in) ; 
	c0 += qpwork._active_part_z.squaredNorm();
	b0 += qpwork._err.tail(qpmodel._n_in).dot(qpwork._active_part_z) ; // [Cdx-dz/mu_in]_act dot [Cx-u+(ze-z_act)/mu]_+ + [Cx-l+(ze-z_act)/mu]--
	// derive z_inact
	qpwork._rhs.tail(qpmodel._n_in).noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() < T(0.) || qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() > T(0.) ).select(qpwork._z_prev, Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	// derive dz_inact
	qpwork._active_part_z.noalias() = (qpwork._primal_residual_in_scaled_up_plus_alphaCdx.array() < T(0.) || qpwork._primal_residual_in_scaled_low_plus_alphaCdx.array() > T(0.) ).select(qpwork._dw_aug.tail(qpmodel._n_in), Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel._n_in));
	b0 += qpwork._rhs.tail(qpmodel._n_in).dot(qpwork._active_part_z);
	c0 += qpwork._rhs.tail(qpmodel._n_in).squaredNorm(); // z_inact squared

	
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


template <typename T>
void initial_guess_LS(
		qp::Qpsettings<T>& qpsettings,
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork
		) {
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

	qpwork._alpha = T(1.);

	T alpha_n(1.);
	T gr_n = line_search::gradient_norm_computation_box(
					qpmodel,
					qpresults,
					qpwork,
					alpha_n) ;
	T gr_interval(1.);
	T alpha_interval(1.);
	T alpha_(0.);

	/////////// STEP 1 ////////////
	// computing the "nodes" alphas which cancel  C.dot(xe+alpha dx) - u,
	// C.dot(xe+alpha dx) - l and ze + alpha dz  /////////////

	qpwork._alphas.clear();

	// 1.1 add solutions of equation z+alpha dz = 0
	
	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (std::abs(qpwork._z_prev(i)) != 0.) {
			alpha_ = -qpwork._z_prev(i) / (qpwork._dw_aug.tail(qpmodel._n_in)(i) + machine_eps);
			if (std::abs(alpha_)< qpsettings._R){
				qpwork._alphas.push_back(alpha_);
			}
		}
	}

	// 1.1 add solutions of equations C(x+alpha dx)-u +ze/mu_in = 0 and C(x+alpha
	// dx)-l +ze/mu_in = 0

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (std::abs(qpwork._Cdx(i)) != 0) {
			alpha_= -qpwork._primal_residual_in_scaled_up(i) / (qpwork._Cdx(i) + machine_eps);
			if (std::abs(alpha_) < qpsettings._R){
				qpwork._alphas.push_back(alpha_);
			}
			alpha_ = -qpwork._primal_residual_in_scaled_low(i) / (qpwork._Cdx(i) + machine_eps);
			if (std::abs(alpha_) < qpsettings._R){
				qpwork._alphas.push_back(alpha_);
			}
		}
	}

	isize n_alpha = qpwork._alphas.size();
	
	if (n_alpha!=0) {
		//////// STEP 2 ////////
		// 2.1/ it sorts alpha nodes

		std::sort (qpwork._alphas.begin(), qpwork._alphas.begin()+n_alpha); 
		qpwork._alphas.erase( std::unique( qpwork._alphas.begin(), qpwork._alphas.begin()+n_alpha), qpwork._alphas.begin()+n_alpha );
		n_alpha = qpwork._alphas.size();
		// 2.2/ for each node active set and associated gradient are computed

		bool first(true);

		for (isize i=0;i<n_alpha;++i) {
			alpha_ = qpwork._alphas[i];
			if (std::abs(alpha_) < T(1.e6)) {
				
				// calcul de la norm du gradient du noeud
				T grad_norm = line_search::gradient_norm_computation_box(
						qpmodel,
						qpresults,
						qpwork,
						alpha_
						);

				if (first){
					alpha_n = alpha_;
					gr_n = grad_norm;
					first = false;
				}else{
					if (grad_norm<gr_n){
						alpha_n = alpha_;
						gr_n = grad_norm;
					}
				} 
			} 
		}
		first = true;
		//////////STEP 3 ////////////
		// 3.1 : define intervals with alphas

		for (isize i = -1; i < n_alpha; ++i) {

			// 3.2 : it derives the mean node (alpha[i]+alpha[i+1])/2
			// the corresponding active sets active_inequalities_u and
			// active_inequalities_l cap ze and dz is derived through function
			// local_saddle_point_box

			if (i == -1){
				alpha_ = qpwork._alphas[0] - T(0.5);
			} else if (i==n_alpha-1){
				alpha_ = qpwork._alphas[n_alpha-1] +  T(0.5);
			} else{
				alpha_ = (qpwork._alphas[i] + qpwork._alphas[i + 1]) * T(0.5);
			}

			// 3.3 on this interval the merit function is a second order
			// polynomial in alpha
			// the function "local_saddle_point_box" derives the exact minimum
			// and corresponding merit function L2 norm (for this minimum)
			T associated_grad_2_norm = line_search::local_saddle_point_box(
					qpmodel,
					qpresults,
					qpwork,
					alpha_
					);

			// 3.4 if the argmin is within the interval [alpha[i],alpha[i+1]] is
			// stores the argmin and corresponding L2 norm

			if (i == -1) {
				if (alpha_ <= qpwork._alphas[0]) {
					
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
			} else if (i == n_alpha-1) {
				if (alpha_ >= qpwork._alphas[n_alpha - 1]) {
					
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
				if (alpha_ <= qpwork._alphas[i + 1] && qpwork._alphas[i] <= alpha_) {
					
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

		if (gr_interval <= gr_n){
			qpwork._alpha = alpha_interval; 
		}else{
			qpwork._alpha = alpha_n;
		}
	}

}


template <typename T>
void correction_guess_LS(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		qp::Qpworkspace<T>& qpwork
		){

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

	auto x_ = qpresults._x;

	qpwork._alpha = T(1.);
	T alpha_(1.);

	qpwork._alphas.clear();

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	for (isize i = 0; i < qpmodel._n_in; i++) {
		if (qpwork._Cdx(i) != 0.) {
			qpwork._alphas.push_back(-qpwork._primal_residual_in_scaled_up(i) / (qpwork._Cdx(i) + machine_eps));
			qpwork._alphas.push_back(-qpwork._primal_residual_in_scaled_low(i) / (qpwork._Cdx(i) + machine_eps));
		}
	}

	isize n_alpha = qpwork._alphas.size();

	if (n_alpha!=0) {
		// 1.2 sort the alphas

		std::sort (qpwork._alphas.begin(), qpwork._alphas.begin()+n_alpha); 
		qpwork._alphas.erase( std::unique( qpwork._alphas.begin(), qpwork._alphas.begin()+n_alpha), qpwork._alphas.begin()+n_alpha );
		n_alpha = qpwork._alphas.size();

		////////// STEP 2 ///////////

		T last_neg_grad = 0;
		T alpha_last_neg = 0;
		T first_pos_grad = 0;
		T alpha_first_pos = 0;

		for (isize i = 0;i<n_alpha;++i){
			alpha_ = qpwork._alphas[i];
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
							qpmodel,
							qpresults,
							qpwork,
							alpha_
							);

					if (gr < T(0)) {
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
		if (last_neg_grad == T(0)) {
			alpha_last_neg = T(0);
			T gr = line_search::gradient_norm_qpalm_box(
					qpmodel,
					qpresults,
					qpwork,
					alpha_last_neg
					);
			last_neg_grad = gr;
		}

		/*
		 * 2.3
		 * the optimal alpha is within the interval
		 * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi'
		 * is an affine function in alpha
		 */
		qpwork._alpha = alpha_last_neg - last_neg_grad *
		                             (alpha_first_pos - alpha_last_neg) /
		                             (first_pos_grad - last_neg_grad);
	}	
}

template <typename T>
void active_set_change(
		qp::Qpdata<T>& qpmodel,
		qp::Qpresults<T>& qpresults,
		Qpworkspace<T>& qpwork
		) {

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

	qpwork._dw_aug.setZero();
	
	isize n_c_f = qpresults._n_c;
	qpwork._new_bijection_map = qpwork._current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set

	T mu_in_inv_neg = -qpresults._mu_in_inv;
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
				
				qpwork._dw_aug.setZero();
				qpwork._dw_aug.head(qpmodel._dim) = qpwork._c_scaled.row(i);
				qpwork._dw_aug(qpmodel._dim + qpmodel._n_eq + n_c_f) = mu_in_inv_neg; // mu stores the inverse of mu
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
	qpwork._dw_aug.setZero();
}

} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS */
