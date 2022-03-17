#ifndef INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS
#define INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS

#include "ldlt/views.hpp"
#include "qp/views.hpp"
#include "qp/QPData.hpp"
#include "qp/QPResults.hpp"
#include "qp/QPWorkspace.hpp"
#include "qp/QPSettings.hpp"
#include <cmath>
#include <iostream>
#include <fstream>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}
namespace line_search {

template <typename T>
auto primal_dual_gradient_norm(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T alpha) -> T {

	/*
	 * the function computes the first derivative of the proximal augmented
	 * lagrangian of the problem at outer step k and inner step l
	 *
	 * phi(alpha) = f(x_l+alpha dx) + rho/2 |x_l + alpha dx - x_k|**2
	 *              + mu_eq/2 (|A(x_l+alpha dx)-d+y_k/mu_eq|**2)
	 *              + mu_eq * nu /2 (|A(x_l+alpha dx)-d+y_k/mu_eq - (y_l+alpha dy)
	 * |**2)
	 *              + mu_in/2 ( | [C(x_l+alpha dx) - u + z_k/mu_in]_+ |**2
	 *                         +| [C(x_l+alpha dx) - l + z_k/mu_in]_- |**2
	 *                         )
	 * 				+ mu_in * nu / 2 ( | [C(x_l+alpha dx) - u + z_k/mu_in]_+ +
	 * [C(x_l+alpha dx) - l + z_k/mu_in]_- - (z+alpha dz)/mu_in |**2 with f(x) =
	 * 0.5 * x^THx + g^Tx phi is a second order polynomial in alpha. Below are
	 * computed its coefficient a0 and b0 in order to compute the desired gradient
	 * a0 * alpha + b0
	 */

	qpwork.primal_residual_in_scaled_up_plus_alphaCdx.noalias() =
			qpwork.primal_residual_in_scaled_up + qpwork.Cdx * alpha;
	qpwork.primal_residual_in_scaled_low_plus_alphaCdx.noalias() =
			qpwork.primal_residual_in_scaled_low + qpwork.Cdx * alpha;

	T a(qpwork.dw_aug.head(qpmodel.dim).dot(qpwork.Hdx) +
	    qpresults.mu_eq * (qpwork.Adx).squaredNorm() +
	    qpresults.rho *
	        qpwork.dw_aug.head(qpmodel.dim)
	            .squaredNorm()); // contains now: a = dx.dot(H.dot(dx)) + rho *
	                             // norm(dx)**2 + (mu_eq) * norm(Adx)**2

	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() =
			qpwork.Adx -
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) * qpresults.mu_eq_inv;
	a += qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).squaredNorm() *
	     qpresults.mu_eq *
	     qpresults
	         .nu; // contains now: a = dx.dot(H.dot(dx)) + rho * norm(dx)**2 +
	              // (mu_eq) * norm(Adx)**2 + nu*mu_eq * norm(Adx-dy*mu_eq_inv)**2
	qpwork.err.head(qpmodel.dim).noalias() =
			qpresults.rho * (qpresults.x - qpwork.x_prev) + qpwork.g_scaled;
	T b(qpresults.x.dot(qpwork.Hdx) +
	    (qpwork.err.head(qpmodel.dim)).dot(qpwork.dw_aug.head(qpmodel.dim)) +
	    qpresults.mu_eq *
	        (qpwork.Adx)
	            .dot(
									qpwork.primal_residual_eq_scaled)); // contains now: b =
	                                                    // dx.dot(H.dot(x) +
	                                                    // rho*(x-xe) +  g)  +
	                                                    // mu_eq * Adx.dot(res_eq)

	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq).noalias() =
			qpwork.primal_residual_eq_scaled - qpresults.y * qpresults.mu_eq_inv;
	b += qpresults.nu * qpresults.mu_eq *
	     qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)
	         .dot(qpwork.rhs.segment(
							 qpmodel.dim,
							 qpmodel.n_eq)); // contains now: b = dx.dot(H.dot(x) + rho*(x-xe)
	                             // +  g)  + mu_eq * Adx.dot(res_eq) + nu*mu_eq *
	                             // (Adx-dy*mu_eq_inv).dot(res_eq-y*mu_eq_inv)

	// derive Cdx_act
	qpwork.err.tail(qpmodel.n_in).noalias() =
			((qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.)) ||
	     (qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.)))
					.select(
							qpwork.Cdx,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));

	a += qpresults.mu_in *
	     qpwork.err.tail(qpmodel.n_in)
	         .squaredNorm(); // contains now: a = dx.dot(H.dot(dx)) + rho *
	                         // norm(dx)**2 + (mu_eq) * norm(Adx)**2 + nu*mu_eq *
	                         // norm(Adx-dy*mu_eq_inv)**2 + mu_in *
	                         // norm(Cdx_act)**2

	// derive vector [Cx-u+ze/mu]_+ + [Cx-l+ze/mu]--
	qpwork.active_part_z.noalias() =
			(qpwork.primal_residual_in_scaled_up_plus_alphaCdx.array() > T(0.))
					.select(
							qpwork.primal_residual_in_scaled_up,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in)) +
			(qpwork.primal_residual_in_scaled_low_plus_alphaCdx.array() < T(0.))
					.select(
							qpwork.primal_residual_in_scaled_low,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));

	b += qpresults.mu_in *
	     qpwork.active_part_z.dot(qpwork.err.tail(
					 qpmodel.n_in)); // contains now: b = dx.dot(H.dot(x) + rho*(x-xe) +
	                         // g)  + mu_eq * Adx.dot(res_eq) + nu*mu_eq *
	                         // (Adx-dy*mu_eq_inv).dot(res_eq-y*mu_eq_inv) + mu_in
	                         // * Cdx_act.dot([Cx-u+ze/mu]_+ + [Cx-l+ze/mu]--)

	// derive Cdx_act - dz/mu_in
	qpwork.err.tail(qpmodel.n_in).noalias() -=
			qpwork.dw_aug.tail(qpmodel.n_in) * qpresults.mu_in_inv;
	// derive [Cx-u+ze/mu]_+ + [Cx-l+ze/mu]-- -z/mu_in
	qpwork.active_part_z.noalias() -= qpresults.z * qpresults.mu_in_inv;

	// contains now a = dx.dot(H.dot(dx)) + rho * norm(dx)**2 + (1./mu_eq) *
	// norm(Adx)**2 + nu/mu_eq * norm(Adx-dy*mu_eq)**2 + 1./mu_in *
	// norm(Cdx_act)**2 + nu/mu_in * norm(Cdx_act-dz/mu_in)**2
	a += qpresults.nu * qpresults.mu_in *
	     qpwork.err.tail(qpmodel.n_in).squaredNorm();
	// contains now b =  dx.dot(H.dot(x) + rho*(x-xe) +  g)  + 1./mu_eq *
	// Adx.dot(res_eq) + nu/mu_eq * (Adx-dy*mu_eq).dot(res_eq-y*mu_eq) + 1./mu_in
	// * Cdx_act.dot([Cx-u+ze/mu]_+ + [Cx-l+ze/mu]--) + nu/mu_in
	// (Cdx_act-dz/mu_in).dot([Cx-u+ze/mu]_+ + [Cx-l+ze/mu]-- - z/mu_in)
	b += qpresults.nu * qpresults.mu_in *
	     qpwork.err.tail(qpmodel.n_in).dot(qpwork.active_part_z);

	return a * alpha + b;
}

template <typename T>
void primal_dual_ls(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		const qp::QPSettings<T>& qpsettings) {

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
	 * phi(alpha) = [proximal primal dual augmented lagrangian of the subproblem
	 * evaluated at x_k + alpha dx, y_k + alpha dy, z_k + alpha dz] using function
	 * "gradient_norm" By construction for alpha = 0, phi'(alpha) <= 0 and
	 * phi'(alpha) goes to infinity with alpha hence it cancels uniquely at one
	 * optimal alpha*
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
	 * using function "gradient_norm"
	 *
	 * 2.3
	 * the optimal alpha is within the interval
	 * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi' is
	 * an affine function in alph
	 * alpha* = alpha_last_neg
	 *        - last_neg_grad * (alpha_first_pos - alpha_last_neg) /
	 *                          (first_pos_grad - last_neg_grad);
	 */

	const T machine_eps = std::numeric_limits<T>::epsilon();

	qpwork.alpha = T(1);
	T alpha_(1.);

	qpwork.alphas.clear();

	///////// STEP 1 /////////
	// 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha
	// dx)-u +ze/mu_in = 0

	// IF NAN DO SOMETHING --> investigate why there is nan and how to fix it
	// (alpha=1?)

	for (isize i = 0; i < qpmodel.n_in; i++) {

		if (qpwork.Cdx(i) != 0.) {
			alpha_ = -qpwork.primal_residual_in_scaled_up(i) /
			         (qpwork.Cdx(i) + machine_eps);
			if (alpha_ > machine_eps) {
				qpwork.alphas.push_back(alpha_);
			}
			alpha_ = -qpwork.primal_residual_in_scaled_low(i) /
			         (qpwork.Cdx(i) + machine_eps);
			if (alpha_ > machine_eps) {
				qpwork.alphas.push_back(alpha_);
			}
		}
	}

	isize n_alpha = qpwork.alphas.size();

	if (n_alpha != 0) {
		// 1.2 sort the alphas

		std::sort(qpwork.alphas.begin(), qpwork.alphas.begin() + n_alpha);
		qpwork.alphas.erase(
				std::unique(qpwork.alphas.begin(), qpwork.alphas.begin() + n_alpha),
				qpwork.alphas.begin() + n_alpha);
		n_alpha = qpwork.alphas.size();

		////////// STEP 2 ///////////

		T last_neg_grad = 0;
		T alpha_last_neg = 0;
		T first_pos_grad = 0;
		T alpha_first_pos = 0;
		for (isize i = 0; i < n_alpha; ++i) {
			alpha_ = qpwork.alphas[i];

			/*
			 * 2.1
			 * For each positive alpha compute the first derivative of
			 * phi(alpha) = [proximal augmented lagrangian of the
			 *               subproblem evaluated at x_k + alpha dx]
			 * using function "gradient_norm"
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
			T gr = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, alpha_);

			if (gr < T(0)) {
				alpha_last_neg = alpha_;
				last_neg_grad = gr;
			} else {
				first_pos_grad = gr;
				alpha_first_pos = alpha_;
				break;
			}
		}

		/*
		 * 2.2
		 * If first_alpha_pos corresponds to the first positive alpha of
		 * previous loop, then do
		 * last_alpha_neg = 0 and last_grad_neg = phi'(0) using function
		 * "gradient_norm"
		 */
		if (last_neg_grad == T(0)) {
			alpha_last_neg = T(0);
			T gr = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, alpha_last_neg);
			last_neg_grad = gr;
		}

		/*
		 * 2.3
		 * the optimal alpha is within the interval
		 * [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi'
		 * is an affine function in alpha
		 */

		if (first_pos_grad == T(0) && alpha_first_pos == T(0)) {
			if (qpsettings.verbose) {
				std::cout << "alpha first pos never updated, try to found one"
									<< std::endl;
			}

			alpha_first_pos = T(1);
			first_pos_grad = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, alpha_first_pos);
			if (qpsettings.verbose) {
				std::cout << "alpha_first_pos " << alpha_first_pos << " first_pos_grad "
									<< first_pos_grad << std::endl;
			}
			for (isize i = 0; i < 8; ++i) {
				if (first_pos_grad > T(0)) {
					break;
				}
				alpha_first_pos *= 10;
				first_pos_grad = line_search::primal_dual_gradient_norm(
						qpmodel, qpresults, qpwork, alpha_first_pos);
				if (qpsettings.verbose) {
					std::cout << "alpha_first_pos " << alpha_first_pos
										<< " first_pos_grad " << first_pos_grad << std::endl;
				}
			}
		}

		qpwork.alpha = alpha_last_neg - last_neg_grad *
		                                    (alpha_first_pos - alpha_last_neg) /
		                                    (first_pos_grad - last_neg_grad);

		if (qpsettings.verbose) {
			T gr_f = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, qpwork.alpha);
			std::cout << "alpha_last_neg " << alpha_last_neg << " last_neg_grad "
								<< last_neg_grad << " alpha_first_pos " << alpha_first_pos
								<< "first_pos_grad " << first_pos_grad << " alpha_final "
								<< qpwork.alpha << " gr_final " << gr_f << std::endl;
		}

	} else {

		T alpha_last_neg(0);
		T last_neg_grad = line_search::primal_dual_gradient_norm(
				qpmodel, qpresults, qpwork, alpha_last_neg);

		T alpha_first_pos(1);
		T first_pos_grad = line_search::primal_dual_gradient_norm(
				qpmodel, qpresults, qpwork, alpha_first_pos);
		for (isize i = 0; i < 4; ++i) {
			if (first_pos_grad > T(0)) {
				break;
			}
			alpha_first_pos *= 10;
			first_pos_grad = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, alpha_first_pos);
		}

		qpwork.alpha = alpha_last_neg - last_neg_grad *
		                                    (alpha_first_pos - alpha_last_neg) /
		                                    (first_pos_grad - last_neg_grad);
		if (qpsettings.verbose) {
			std::cout << " try finding positive grad " << std::endl;
			T gr_f = line_search::primal_dual_gradient_norm(
					qpmodel, qpresults, qpwork, qpwork.alpha);
			std::cout << "alpha_last_neg " << alpha_last_neg << " last_neg_grad "
								<< last_neg_grad << " alpha_first_pos " << alpha_first_pos
								<< "first_pos_grad " << first_pos_grad << " alpha_final "
								<< qpwork.alpha << " gr_final " << gr_f << std::endl;
		}
	}
}

template <typename T>
void active_set_change(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		QPWorkspace<T>& qpwork) {

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

	qpwork.dw_aug.setZero();

	isize n_c_f = qpresults.n_c;
	qpwork.new_bijection_map = qpwork.current_bijection_map;

	// suppression pour le nouvel active set, ajout dans le nouvel unactive set

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};

	T mu_in_inv_neg = -qpresults.mu_in_inv;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		if (qpwork.current_bijection_map(i) < qpresults.n_c) {
			if (!qpwork.active_inequalities(i)) {
				// delete current_bijection_map(i)

				isize indices[] = {
						qpwork.new_bijection_map(i) + qpmodel.dim + qpmodel.n_eq,
				};

				qpwork.ldl.delete_at(indices, 1, stack);

				for (isize j = 0; j < qpmodel.n_in; j++) {
					if (qpwork.new_bijection_map(j) > qpwork.new_bijection_map(i)) {
						qpwork.new_bijection_map(j) -= 1;
					}
				}
				n_c_f -= 1;
				qpwork.new_bijection_map(i) = qpmodel.n_in - 1;
			}
		}
	}

	// ajout au nouvel active set, suppression pour le nouvel unactive set

	auto _planned_to_add =
			stack.make_new_for_overwrite(veg::Tag<isize>{}, qpmodel.n_in).unwrap();
	auto planned_to_add = _planned_to_add.ptr_mut();

	isize planned_to_add_count = 0;

	isize n_c = n_c_f;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		if (qpwork.active_inequalities(i)) {
			if (qpwork.new_bijection_map(i) >= n_c_f) {
				// add at the end
				planned_to_add[planned_to_add_count] = i;
				++planned_to_add_count;

				// qpwork.dw_aug.setZero();
				// qpwork.dw_aug.head(qpmodel.dim) = qpwork.C_scaled.row(i);
				// qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + n_c_f) =
				// 		mu_in_inv_neg; // mu stores the inverse of mu

				// isize insert_dim = n_c_f + 1 + qpmodel.n_eq + qpmodel.dim;
				// qpwork.ldl.insert_block_at(
				// 		qpmodel.n_eq + qpmodel.dim + n_c_f,
				// 		qpwork.dw_aug.head(insert_dim),
				// 		stack);

				for (isize j = 0; j < qpmodel.n_in; j++) {
					if (qpwork.new_bijection_map(j) < qpwork.new_bijection_map(i) &&
					    qpwork.new_bijection_map(j) >= n_c_f) {
						qpwork.new_bijection_map(j) += 1;
					}
				}
				qpwork.new_bijection_map(i) = n_c_f;
				n_c_f += 1;
			}
		}
	}
	{
		isize n = qpmodel.dim;
		isize n_eq = qpmodel.n_eq;
		LDLT_TEMP_MAT_UNINIT(
				T, new_cols, n + n_eq + n_c_f, planned_to_add_count, stack);

		for (isize k = 0; k < planned_to_add_count; ++k) {
			isize index = planned_to_add[k];
			auto col = new_cols.col(k);
			col.head(n) = (qpwork.C_scaled.row(index));
			col.tail(n_eq + n_c_f).setZero();
			col[n + n_eq + n_c + k] = -qpresults.mu_in_inv;
		}
		qpwork.ldl.insert_block_at(n + n_eq + n_c, new_cols, stack);
	}

	qpresults.n_c = n_c_f;
	qpwork.current_bijection_map = qpwork.new_bijection_map;
	qpwork.dw_aug.setZero();
}

} // namespace line_search
} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_LINE_SEARCH_HPP_2TUXO5DFS */
