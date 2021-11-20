#ifndef INRIA_LDLT_LINE_SEARCH_QPALM_HPP_2TUXO5DFS
#define INRIA_LDLT_LINE_SEARCH_QPALM_HPP_2TUXO5DFS

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
namespace line_search_qpalm {

template <typename T>
auto gradient_norm_qpalm_box_(
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> dx,
		VectorView<T> mu_,
		T rho,
		T alpha,
		VectorView<T> Hdx_,
		VectorView<T> g_,
		VectorView<T> Adx_,
		VectorView<T> residual_in_y_,
		VectorView<T> residual_in_z_u_,
		VectorView<T> residual_in_z_l_,
		VectorView<T> Cdx_,
		isize n_in,
		isize n_eq) -> T {

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

	auto mu = mu_.to_eigen();
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
	auto muAdx = (mu.topRows(n_eq).array() * Adx.array() ).matrix() ; 

	T a(dx_.dot(Hdx) + muAdx.dot(Adx) + rho * dx_.squaredNorm());
	T b(x_.dot(Hdx) + (rho * (x_ - xe_) + g).dot(dx_) + muAdx.dot(residual_in_y));

	for (isize k = 0; k < n_in; ++k) {

		if (tmp_u(k) > T(0.)) {

			a += mu(n_eq+k) * pow(Cdx(k), T(2));
			b += mu(n_eq+k) * Cdx(k) * residual_in_z_u(k);

		} else if (tmp_l(k) < T(0.)) {

			a += mu(n_eq+k) * pow(Cdx(k), T(2));
			b += mu(n_eq+k) * Cdx(k) * residual_in_z_l(k);
		}
	}

	return a * alpha + b;
}

template <typename T>
auto correction_guess_LS_QPALM(
		VectorView<T> Hdx_,
		VectorView<T> dx,
		VectorView<T> g,
		VectorView<T>  Adx_,
		VectorView<T> Cdx_,
		VectorView<T>  residual_in_y_,
		VectorView<T>  residual_in_z_u_,
		VectorView<T>  residual_in_z_l_,
		VectorView<T> x,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorView<T> mu_,
		T rho,
		isize n_in,
		isize n_eq) -> T {

	auto Hdx = Hdx_.to_eigen();
	auto Adx = Adx_.to_eigen();
	auto Cdx = Cdx_.to_eigen();
	auto mu = mu_.to_eigen();
	auto residual_in_y = residual_in_y_.to_eigen();
	auto residual_in_z_u = residual_in_z_u_.to_eigen();
	auto residual_in_z_l = residual_in_z_l_.to_eigen();
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

	T alpha = T(1);

	std::list<T> alphas = {};

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	for (isize i = 0; i < n_in; i++) {
		if (Cdx(i) != 0) {
			alphas.push_back(-residual_in_z_u(i) / (Cdx(i) + machine_eps));
		}
		if (Cdx(i) != 0) {
			alphas.push_back(-residual_in_z_l(i) / (Cdx(i) + machine_eps));
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
					T gr = line_search_qpalm::gradient_norm_qpalm_box_(
							x,
							xe,
							dx,
							mu_,
							rho,
							a,
							VectorView<T>{from_eigen, Hdx},
							g,
							VectorView<T>{from_eigen, Adx},
							VectorView<T>{from_eigen, residual_in_y},
							VectorView<T>{from_eigen, residual_in_z_u},
							VectorView<T>{from_eigen, residual_in_z_l},
							VectorView<T>{from_eigen, Cdx},
							n_in,
							n_eq);
					if (gr < T(0)) {
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
			T gr = line_search_qpalm::gradient_norm_qpalm_box_(
							x,
							xe,
							dx,
							mu_,
							rho,
							alpha_last_neg,
							VectorView<T>{from_eigen, Hdx},
							g,
							VectorView<T>{from_eigen, Adx},
							VectorView<T>{from_eigen, residual_in_y},
							VectorView<T>{from_eigen, residual_in_z_u},
							VectorView<T>{from_eigen, residual_in_z_l},
							VectorView<T>{from_eigen, Cdx},
							n_in,
							n_eq);
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


template <typename T>
void active_set_change_QPALM(
		VectorView<bool> new_active_set_,
		VectorViewMut<isize> current_bijection_map_,
		isize& n_c,
		isize n_in,
		isize dim,
		isize n_eq,
		ldlt::Ldlt<T>& ldl,
		qp::QpViewBox<T> qp,
		VectorView<T> mu_,
		T rho,
		T max_rank_update,
		T max_rank_update_fraction,
		isize& nb_enter,
		isize& nb_leave,
		bool& VERBOSE
		) {
	
	auto mu = mu_.to_eigen();
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

	for (isize i = 0; i < n_in; i++) {
		if (current_bijection_map(i) < n_c) {
			if (!new_active_set(i)) {
				nb_leave +=1;
			}
		}
	}
	for (isize i = 0; i < n_in; i++) {
		if (new_active_set(i)) {
			if (new_bijection_map(i) >= n_c_f) {
				nb_enter +=1;
			}
		}
	}
	//std::cout << "test " << test << " n_leaving_constraint " << n_leaving_constraint << " n_entering_constraint " << n_entering_constraint << " std::min(max_rank_update,max_rank_update_fraction * T(dim+n_eq+n_in)) " << std::min(max_rank_update,max_rank_update_fraction * T(dim+n_eq+n_in)) << std::endl;
	if (T(nb_enter + nb_leave) <= std::min(max_rank_update,max_rank_update_fraction * T(dim+n_eq+n_in))) {

		// suppression pour le nouvel active set, ajout dans le nouvel unactive set
		for (isize i = 0; i < n_in; i++) {
			if (current_bijection_map(i) < n_c) {
				if (!new_active_set(i)) {
					ldl.delete_at(new_bijection_map(i) + dim + n_eq);
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
						row(dim + n_eq + n_c_f) = -T(1) / mu(n_eq + i);
						ldl.insert_at(n_eq + dim + n_c_f, row);
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

	} else{

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
		n_c = n_c_f;
		current_bijection_map_.to_eigen() = new_bijection_map;

		[&]{
			LDLT_MULTI_WORKSPACE_MEMORY(
				(_htot,Uninit, Mat(dim+n_eq+n_c, dim+n_eq+n_c),LDLT_CACHELINE_BYTES, T)
			);
			
			auto Htot = _htot.to_eigen().eval();
			Htot.setZero();
			
			Htot.topLeftCorner(dim, dim) = qp.H.to_eigen();
			for (isize i = 0; i < dim; ++i) {
				Htot(i, i) += rho; 
			}
			
			Htot.block(0,dim,dim,n_eq) = qp.A.to_eigen().transpose();
			Htot.block(dim,0,n_eq,dim) = qp.A.to_eigen();
			{
				for (isize i = 0; i < n_eq; ++i) {
					Htot(dim + i, dim + i) = -T(1) / mu( i);;
				}
			}
			for (isize i = 0; i< n_in ; ++i){
				isize j = new_bijection_map(i);
				if (j<n_c){
					Htot.block(j+dim+n_eq,0,1,dim) = qp.C.to_eigen().row(i) ; 
					Htot.block(0,j+dim+n_eq,dim,1) = qp.C.to_eigen().transpose().col(i)  ; 
					Htot(dim + n_eq + j, dim + n_eq + j) = -T(1) / mu(n_eq + i);
				}
			}
			ldl.factorize(Htot);
			if (VERBOSE){
				std::cout << "error " <<  qp::infty_norm(Htot-ldl.reconstructed_matrix()) << std::endl; 
			}
		}();
	}

}

} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_LINE_SEARCH_QPALM_HPP_2TUXO5DFS */