#ifndef INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS

#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/proxqp/line_search.hpp"
#include <cmath>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>

#include <dense-ldlt/ldlt.hpp>

template <typename Derived>
void save_data(
		const std::string& filename, const Eigen::MatrixBase<Derived>& mat) {
	// https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static Eigen::IOFormat CSVFormat(
			Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << mat.format(CSVFormat);
		file.close();
	}
}

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

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
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T rho_new) {

	qpwork.dw_aug.setZero();
	qpwork.kkt.diagonal().head(qpmodel.dim).array() += rho_new - qpresults.rho;
	qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
			-qpresults.mu_eq_inv;
	{
		veg::dynstack::DynStackMut stack{
				veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
		qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));
	}

	for (isize j = 0; j < qpresults.n_c; ++j) {
		for (isize i = 0; i < qpmodel.n_in; ++i) {
			if (j == qpwork.current_bijection_map(i)) {
				qpwork.dw_aug.head(qpmodel.dim) = qpwork.C_scaled.row(i);
				qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) =
						-qpresults.mu_in_inv; // mu_in stores the inverse of mu_in
				{
					isize insert_dim = qpmodel.dim + qpmodel.n_eq + qpresults.n_c;
					veg::dynstack::DynStackMut stack{
							veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
					qpwork.ldl.insert_at(
							qpmodel.n_eq + qpmodel.dim + j,
							qpwork.dw_aug.head(insert_dim),
							LDLT_FWD(stack));
				}
				qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) = T(0);
			}
		}
	}
	qpwork.dw_aug.setZero();
}

template <typename T>
void mu_update(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T mu_eq_new_inv,
		T mu_in_new_inv) {
	T diff = 0;

	isize total_dim = qpmodel.dim + qpmodel.n_eq + qpresults.n_c;
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};

	qpwork.dw_aug.head(qpmodel.dim + qpmodel.n_eq + qpresults.n_c).setZero();
	if (qpmodel.n_eq > 0) {
		diff = qpresults.mu_eq_inv - mu_eq_new_inv; // mu stores the inverse of mu

		for (isize i = 0; i < qpmodel.n_eq; i++) {
			qpwork.dw_aug(qpmodel.dim + i) = T(1);
			qpwork.ldl.rank_one_update(
					qpwork.dw_aug.head(total_dim), diff, LDLT_FWD(stack));
			qpwork.dw_aug(qpmodel.dim + i) = T(0);
		}
	}
	if (qpresults.n_c > 0) {
		diff = qpresults.mu_in_inv - mu_in_new_inv; // mu stores the inverse of mu
		for (isize i = 0; i < qpresults.n_c; i++) {
			qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i) = T(1);
			qpwork.ldl.rank_one_update(
					qpwork.dw_aug.head(total_dim), diff, LDLT_FWD(stack));
			qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i) = T(0);
		}
	}
}

template <typename T>
void iterative_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		isize inner_pb_dim) {

	qpwork.err.head(inner_pb_dim).noalias() = qpwork.rhs.head(inner_pb_dim);
	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.H_scaled * qpwork.dw_aug.head(qpmodel.dim);
	qpwork.err.head(qpmodel.dim).noalias() -=
			qpresults.rho * qpwork.dw_aug.head(qpmodel.dim);
	qpwork.err.head(qpmodel.dim).noalias() -=
			qpwork.A_scaled.transpose() *
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.n_c) {
			qpwork.err.head(qpmodel.dim).noalias() -=
					qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
					qpwork.C_scaled.row(i);
			qpwork.err(qpmodel.dim + qpmodel.n_eq + j) -=
					(qpwork.C_scaled.row(i).dot(qpwork.dw_aug.head(qpmodel.dim)) -
			     qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) *
			         qpresults.mu_in_inv); // mu stores the inverse of mu
		}
	}
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() -=
			qpwork.A_scaled *
			qpwork.dw_aug.head(qpmodel.dim); // mu stores the inverse of mu
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq) +=
			qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) *
			qpresults.mu_eq_inv; // mu stores the inverse of mu
}

template <typename T>
void iterative_solve_with_permut_fact( //
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps,
		isize inner_pb_dim) {

	qpwork.err.setZero();
	i32 it = 0;
	i32 it_stability = 0;

	qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut, qpwork.ldl_stack.as_mut()};
	qpwork.ldl.solve_in_place(qpwork.dw_aug.head(inner_pb_dim), LDLT_FWD(stack));

	qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

	++it;
	T preverr = infty_norm(qpwork.err.head(inner_pb_dim));
	if (qpsettings.verbose) {
		std::cout << "infty_norm(res) "
							<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
	}
	while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

		if (it >= qpsettings.nb_iterative_refinement) {
			break;
		}

		++it;
		qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), LDLT_FWD(stack));
		qpwork.dw_aug.head(inner_pb_dim).noalias() += qpwork.err.head(inner_pb_dim);

		qpwork.err.head(inner_pb_dim).setZero();
		qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
			it_stability += 1;

		} else {
			it_stability = 0;
		}
		if (it_stability == 2) {
			break;
		}
		preverr = infty_norm(qpwork.err.head(inner_pb_dim));

		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
	}

	if (infty_norm(qpwork.err.head(inner_pb_dim)) >=
	    std::max(eps, qpsettings.eps_refact)) {
		{
			/*
			LDLT_MULTI_WORKSPACE_MEMORY(
			  (_htot,Uninit, Mat(qpmodel.dim+qpmodel.n_eq+qpresults.n_c,
			qpmodel.dim+qpmodel.n_eq+qpresults.n_c),LDLT_CACHELINE_BYTES, T)
			  );
			auto Htot = _htot.to_eigen().eval();
			Htot.setZero();
			qpwork.kkt.diagonal().segment(qpmodel.dim,qpmodel.n_eq).array() =
			-qpresults.mu_eq_inv; Htot.topLeftCorner(qpmodel.dim+qpmodel.n_eq,
			qpmodel.dim+qpmodel.n_eq) = qpwork.kkt;
			Htot.diagonal().segment(qpmodel.dim+qpmodel.n_eq,qpresults.n_c).array() =
			-qpresults.mu_in_inv; for (isize i = 0; i< qpmodel.n_in ; ++i){
			    isize j = qpwork.current_bijection_map(i);
			    if (j<qpresults.n_c){
			      Htot.block(j+qpmodel.dim+qpmodel.n_eq,0,1,qpmodel.dim) =
			qpwork.C_scaled.row(i) ;
			      Htot.block(0,j+qpmodel.dim+qpmodel.n_eq,qpmodel.dim,1) =
			qpwork.C_scaled.transpose().col(i) ;
			    }
			}
			qpwork.ldl.factorize(Htot);
			*/
			qpwork.dw_aug.setZero();
			qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
					-qpresults.mu_eq_inv;
			qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));

			for (isize j = 0; j < qpresults.n_c; ++j) {
				for (isize i = 0; i < qpmodel.n_in; ++i) {
					if (j == qpwork.current_bijection_map(i)) {
						qpwork.dw_aug.head(qpmodel.dim) = qpwork.C_scaled.row(i);
						qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) =
								-qpresults.mu_in_inv; // mu_in stores the inverse of mu_in
						{
							isize insert_dim = qpmodel.dim + qpmodel.n_eq + qpresults.n_c;
							qpwork.ldl.insert_at(
									qpmodel.n_eq + qpmodel.dim + j,
									qpwork.dw_aug.head(insert_dim),
									LDLT_FWD(stack));
						}
						qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + j) = T(0);
					}
				}
			}
			qpwork.dw_aug.setZero();

			// std::cout << " ldl.reconstructed_matrix() - Htot " <<
			// infty_norm(qpwork.ldl.reconstructed_matrix() - Htot)<< std::endl;
		}
		it = 0;
		it_stability = 0;

		qpwork.dw_aug.head(inner_pb_dim) = qpwork.rhs.head(inner_pb_dim);
		qpwork.ldl.solve_in_place(
				qpwork.dw_aug.head(inner_pb_dim), LDLT_FWD(stack));

		qp::detail::iterative_residual<T>(qpmodel, qpresults, qpwork, inner_pb_dim);

		preverr = infty_norm(qpwork.err.head(inner_pb_dim));
		++it;
		if (qpsettings.verbose) {
			std::cout << "infty_norm(res) "
								<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
		}
		while (infty_norm(qpwork.err.head(inner_pb_dim)) >= eps) {

			if (it >= qpsettings.nb_iterative_refinement) {
				break;
			}
			++it;
			qpwork.ldl.solve_in_place(qpwork.err.head(inner_pb_dim), LDLT_FWD(stack));
			qpwork.dw_aug.head(inner_pb_dim).noalias() +=
					qpwork.err.head(inner_pb_dim);

			qpwork.err.head(inner_pb_dim).setZero();
			qp::detail::iterative_residual<T>(
					qpmodel, qpresults, qpwork, inner_pb_dim);

			if (infty_norm(qpwork.err.head(inner_pb_dim)) > preverr) {
				it_stability += 1;

			} else {
				it_stability = 0;
			}
			if (it_stability == 2) {
				break;
			}
			preverr = infty_norm(qpwork.err.head(inner_pb_dim));

			if (qpsettings.verbose) {
				std::cout << "infty_norm(res) "
									<< qp::infty_norm(qpwork.err.head(inner_pb_dim)) << std::endl;
			}
		}
	}
	qpwork.rhs.head(inner_pb_dim).setZero();
}

template <typename T>
void bcl_update(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& primal_feasibility_lhs,
		T& bcl_eta_ext,
		T& bcl_eta_in,

		T bcl_eta_ext_init,
		T eps_in_min,

		T& new_bcl_mu_in,
		T& new_bcl_mu_eq,
		T& new_bcl_mu_in_inv,
		T& new_bcl_mu_eq_inv

) {

	if (primal_feasibility_lhs <= bcl_eta_ext) {
		if (qpsettings.verbose) {
			std::cout << "good step" << std::endl;
		}
		bcl_eta_ext = bcl_eta_ext * pow(qpresults.mu_in_inv, qpsettings.beta_bcl);
		bcl_eta_in = max2(bcl_eta_in * qpresults.mu_in_inv, eps_in_min);
	} else {
		if (qpsettings.verbose) {
			std::cout << "bad step" << std::endl;
		}

		qpresults.y = qpwork.y_prev;
		qpresults.z = qpwork.z_prev;

		new_bcl_mu_in = std::min(
				qpresults.mu_in * qpsettings.mu_update_factor, qpsettings.mu_max_in);
		new_bcl_mu_eq = std::min(
				qpresults.mu_eq * qpsettings.mu_update_factor, qpsettings.mu_max_eq);
		new_bcl_mu_in_inv = max2(
				qpresults.mu_in_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_in_inv); // mu stores the inverse of mu
		new_bcl_mu_eq_inv = max2(
				qpresults.mu_eq_inv * qpsettings.mu_update_inv_factor,
				qpsettings.mu_max_eq_inv); // mu stores the inverse of mu
		bcl_eta_ext =
				bcl_eta_ext_init * pow(new_bcl_mu_in_inv, qpsettings.alpha_bcl);
		bcl_eta_in = max2(new_bcl_mu_in_inv, eps_in_min);
	}
}

template <typename T>
void global_primal_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& primal_feasibility_lhs,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T& primal_feasibility_eq_lhs,
		T& primal_feasibility_in_lhs) {

	qpwork.primal_residual_eq_scaled.noalias() = qpwork.A_scaled * qpresults.x;
	qpwork.primal_residual_in_scaled_up.noalias() = qpwork.C_scaled * qpresults.x;

	qpwork.ruiz.unscale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
	primal_feasibility_eq_rhs_0 = infty_norm(qpwork.primal_residual_eq_scaled);
	qpwork.ruiz.unscale_primal_residual_in_place_in(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_in_scaled_up});
	primal_feasibility_in_rhs_0 = infty_norm(qpwork.primal_residual_in_scaled_up);

	qpwork.primal_residual_in_scaled_low.noalias() =
			detail::positive_part(qpwork.primal_residual_in_scaled_up - qpmodel.u) +
			detail::negative_part(qpwork.primal_residual_in_scaled_up - qpmodel.l);
	qpwork.primal_residual_eq_scaled -= qpmodel.b;

	primal_feasibility_in_lhs = infty_norm(qpwork.primal_residual_in_scaled_low);
	primal_feasibility_eq_lhs = infty_norm(qpwork.primal_residual_eq_scaled);
	primal_feasibility_lhs =
			max2(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	qpwork.ruiz.scale_primal_residual_in_place_eq(
			VectorViewMut<T>{from_eigen, qpwork.primal_residual_eq_scaled});
}

template <typename T>
void global_dual_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T& dual_feasibility_lhs,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		T& dual_feasibility_rhs_3) {

	qpwork.dual_residual_scaled = qpwork.g_scaled;
	qpwork.CTz.noalias() = qpwork.H_scaled * qpresults.x;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_0 = infty_norm(qpwork.CTz);
	qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * qpresults.y;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_1 = infty_norm(qpwork.CTz);

	qpwork.CTz.noalias() = qpwork.C_scaled.transpose() * qpresults.z;
	qpwork.dual_residual_scaled += qpwork.CTz;
	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.CTz});
	dual_feasibility_rhs_3 = infty_norm(qpwork.CTz);

	qpwork.ruiz.unscale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});

	dual_feasibility_lhs = infty_norm(qpwork.dual_residual_scaled);

	qpwork.ruiz.scale_dual_residual_in_place(
			VectorViewMut<T>{from_eigen, qpwork.dual_residual_scaled});
};

template <typename T>
T compute_primal_dual_residual(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	/*
	qpwork.primal_residual_in_scaled_up_plus_alphaCdx.noalias()
	= qp::detail::positive_part(qpwork.primal_residual_in_scaled_up) ; //
	[Cx-u+z_prev/mu_in]+
	qpwork.primal_residual_in_scaled_low_plus_alphaCdx.noalias()
	= qp::detail::negative_part(qpwork.primal_residual_in_scaled_low); //
	[Cx-l+z_prev/mu_in]-
	qpwork.active_set_up.array()  = ( qpresults.z.array() == T(0));
	qpwork.active_part_z.noalias() =
	(qpwork.active_set_up).select(qpwork.primal_residual_in_scaled_up_plus_alphaCdx,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))  ; T err =
	max2(err,infty_norm(qpwork.active_part_z)); //  ||[Cx-u+z_prev/mu_in]+_[z=0]||
	std::cout << " ||[Cx-u+z_prev/mu_in]+_[z=0]|| " <<
	infty_norm(qpwork.active_part_z)  << std::endl; qpwork.active_part_z.noalias()
	=
	(qpwork.active_set_up).select(qpwork.primal_residual_in_scaled_low_plus_alphaCdx,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))  ; std::cout << "
	||[[Cx-l+z_prev/mu_in]-]+_[z=0]|| " << infty_norm(qpwork.active_part_z)  <<
	std::endl; err = max2(err,infty_norm(qpwork.active_part_z)); // max of
	previous and ||[[Cx-l+z_prev/mu_in]-]+_[z=0]||
	qpwork.primal_residual_in_scaled_up.noalias() -=
	(qpresults.z*qpresults.mu_in_inv);  // contains now Cx-u-(z-z_prev)/mu
	qpwork.primal_residual_in_scaled_low.noalias() -=
	(qpresults.z*qpresults.mu_in_inv) ; // contains now  Cx-l-(z-z_prev)/mu T
	prim_eq_e = infty_norm(qpwork.primal_residual_eq_scaled) ;  //
	||Ax-b-(y-y_prev)/mu|| std::cout<< " prim_eq_e " << prim_eq_e<< std::endl; err
	= max2(err,prim_eq_e); qpwork.dual_residual_scaled.noalias() +=
	(qpwork.C_scaled.transpose()*qpresults.z);  // contains now Hx + rho(x-xprev)
	+ g + Aty + Ctz T dual_e = infty_norm(qpwork.dual_residual_scaled); std::cout
	<< "dual_e " << dual_e << std::endl; err = max2(err,dual_e);
	/// typo should compute || [Cx-l+z_k/mu_in]-|| + || [Cx-u+z_k/mu_in]+||
	//qpwork.primal_residual_in_scaled_up_plus_alphaCdx.noalias()
	//= qp::detail::positive_part(qpwork.primal_residual_in_scaled_up)
	//+ qp::detail::negative_part(qpwork.primal_residual_in_scaled_low);
	qpwork.active_set_up.array()  = ( qpresults.z.array() > T(0));
	qpwork.active_set_low.array() = ( qpresults.z.array() < T(0));
	//qpwork.active_part_z.noalias()
	//= (qpwork.active_set_up).select(qpwork.primal_residual_in_scaled_up,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))
	//+ (qpwork.active_set_low).select(qpwork.primal_residual_in_scaled_low,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))
	//+ (!qpwork.active_set_low.array() &&
	!qpwork.active_set_up.array()).select(qpwork.primal_residual_in_scaled_up_plus_alphaCdx,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in));
	//err = max2(err,infty_norm(qpwork.active_part_z));
	qpwork.active_part_z.noalias()
	= (qpwork.active_set_up).select(qpwork.primal_residual_in_scaled_up,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))
	+ (qpwork.active_set_low).select(qpwork.primal_residual_in_scaled_low,
	Eigen::Matrix<T,Eigen::Dynamic,1>::Zero(qpmodel.n_in))  ; // contains
	[Cx-u-(z-z_prev)/mu]_[z>0] + [Cx-l-(z-z_prev)/mu]_[z<0] + 0_[z==0] err =
	max2(err,infty_norm(qpwork.active_part_z)); std::cout << "||
	[Cx-u-(z-z_prev)/mu]_[z>0] + [Cx-l-(z-z_prev)/mu]_[z<0] + 0_[z==0] || " <<
	infty_norm(qpwork.active_part_z) << std::endl;
	*/

	qpwork.active_set_up.array() = qpwork.primal_residual_in_scaled_up.array() >=
	                               T(0); // Cx-u+z_prev/mu_in>=0
	qpwork.active_set_low.array() =
			qpwork.primal_residual_in_scaled_low.array() <=
			T(0); // Cx-l+z_prev/mu_in<=0
	qpwork.active_inequalities.noalias() =
			qpwork.active_set_up || qpwork.active_set_low;
	qpwork.primal_residual_in_scaled_up.noalias() -=
			(qpresults.z * qpresults.mu_in_inv); // contains now Cx-u-(z-z_prev)/mu
	qpwork.primal_residual_in_scaled_low.noalias() -=
			(qpresults.z * qpresults.mu_in_inv); // contains now  Cx-l-(z-z_prev)/mu

	qpwork.active_part_z.noalias() =
			(qpwork.active_set_up)
					.select(
							qpwork.primal_residual_in_scaled_up,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in)) +
			(qpwork.active_set_low)
					.select(
							qpwork.primal_residual_in_scaled_low,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in)) +
			(!qpwork.active_inequalities.array())
					.select(
							qpresults.z,
							Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));

	T err = infty_norm(qpwork.active_part_z);
	T prim_eq_e =
			infty_norm(qpwork.primal_residual_eq_scaled); // ||Ax-b-(y-y_prev)/mu||
	err = max2(err, prim_eq_e);
	qpwork.dual_residual_scaled.noalias() +=
			(qpwork.C_scaled.transpose() *
	     qpresults.z); // contains now Hx + rho(x-xprev) + g + Aty + Ctz
	T dual_e = infty_norm(qpwork.dual_residual_scaled);
	err = max2(err, dual_e);

	return err;
}

template <typename T>
T compute_inner_loop_saddle_point(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	qpwork.active_part_z.noalias() =
			qp::detail::positive_part(qpwork.primal_residual_in_scaled_up) +
			qp::detail::negative_part(qpwork.primal_residual_in_scaled_low) -
			qpresults.z * qpresults.mu_in_inv; // contains now : [Cx-u+z_prev/mu_in]+
	                                       // + [Cx-l+z_prev/mu_in]- - z/mu_in

	T err = infty_norm(qpwork.active_part_z);
	qpwork.err.segment(qpmodel.dim, qpmodel.n_eq).noalias() =
			qpwork.primal_residual_eq_scaled -
			qpresults.y * qpresults.mu_eq_inv; // contains now Ax-b-(y-y_prev)/mu
	T prim_eq_e = infty_norm(
			qpwork.err.segment(qpmodel.dim, qpmodel.n_eq)); // ||Ax-b-(y-y_prev)/mu||
	err = max2(err, prim_eq_e);
	T dual_e =
			infty_norm(qpwork.dual_residual_scaled); // contains ||Hx + rho(x-xprev) +
	                                             // g + Aty + Ctz||
	err = max2(err, dual_e);

	return err;
}

template <typename T>
void newton_step(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps) {

	qpwork.active_set_up.array() =
			(qpwork.primal_residual_in_scaled_up.array() > 0);
	qpwork.active_set_low.array() =
			(qpwork.primal_residual_in_scaled_low.array() < 0);
	qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
	isize numactive_inequalities = qpwork.active_inequalities.count();
	isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
	qpwork.rhs.setZero();
	qpwork.dw_aug.setZero();
	qpwork.rhs.head(qpmodel.dim) -= qpwork.dual_residual_scaled;

	qp::line_search::active_set_change(qpmodel, qpresults, qpwork);

	iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			eps,
			inner_pb_dim);
}

template <typename T>
void primal_dual_semi_smooth_newton_step(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps) {

	/* MUST BE
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+1./mu_eq (y_prev)
	 *  primal_residual_in_scaled_up = Cx-u+1./mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+1./mu_in(z_prev)
	 */

	qpwork.active_set_up.array() =
			(qpwork.primal_residual_in_scaled_up.array() >= 0);
	qpwork.active_set_low.array() =
			(qpwork.primal_residual_in_scaled_low.array() <= 0);
	qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
	isize numactive_inequalities = qpwork.active_inequalities.count();

	// std::cout << "numactive_inequalities " << numactive_inequalities<<
	// std::endl;
	isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
	qpwork.rhs.setZero();
	qpwork.dw_aug.setZero();

	qp::line_search::active_set_change(qpmodel, qpresults, qpwork);

	qpwork.rhs.head(qpmodel.dim).noalias() = -qpwork.dual_residual_scaled;
	// std::cout << " qpwork.rhs head before activation " <<
	// qpwork.rhs.head(qpmodel.dim) << std::endl;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq).noalias() =
			-qpwork.primal_residual_eq_scaled + qpresults.y * qpresults.mu_eq_inv;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.n_c) {
			if (qpwork.active_set_up(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_up(i) +
						qpresults.z(i) * qpresults.mu_in_inv;
			} else if (qpwork.active_set_low(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_low(i) +
						qpresults.z(i) * qpresults.mu_in_inv;
			}
		} else {
			qpwork.rhs.head(qpmodel.dim).noalias() +=
					qpresults.z(i) *
					qpwork.C_scaled.row(i); // unactive unrelevant columns
		}
	}
	// std::cout << " qpwork.rhs " << qpwork.rhs<< std::endl;

	iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			eps,
			inner_pb_dim);

	// use active_part_z as a temporary variable to derive unpermutted dz step
	for (isize j = 0; j < qpmodel.n_in; ++j) {
		isize i = qpwork.current_bijection_map(j);
		if (i < qpresults.n_c) {
			qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
		} else {
			qpwork.active_part_z(j) = -qpresults.z(j);
		}
	}
	qpwork.dw_aug.tail(qpmodel.n_in) = qpwork.active_part_z;

	// std::cout << "primal dual newton step " << qpwork.dw_aug << std::endl;
}

template <typename T>
T initial_guess(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		VectorViewMut<T> ze,
		T eps_int) {
	if (qpmodel.n_in != 0) {
		qpwork.ruiz.unscale_dual_in_place_in(ze);
		qpwork.primal_residual_in_scaled_up.noalias() +=
				(ze.to_eigen() *
		     qpresults.mu_in_inv); // contains now unscaled(Cx+ze/mu_in)
		qpwork.primal_residual_in_scaled_low = qpwork.primal_residual_in_scaled_up;
		qpwork.primal_residual_in_scaled_up -= qpmodel.u;
		qpwork.primal_residual_in_scaled_low -= qpmodel.l;
		qpwork.active_set_up.array() =
				(qpwork.primal_residual_in_scaled_up.array() >=
		     0.); // Cx-u + zk/mu >= 0
		qpwork.active_set_low.array() =
				(qpwork.primal_residual_in_scaled_low.array() <=
		     0.); // Cx-l + zk/mu <= 0 --> NB : disjoint if both == 0 as by
		          // assumption u>l
		qpwork.active_inequalities = qpwork.active_set_up || qpwork.active_set_low;
		qpwork.primal_residual_in_scaled_up.noalias() -=
				(ze.to_eigen() * qpresults.mu_in_inv);
		qpwork.primal_residual_in_scaled_low.noalias() -=
				(ze.to_eigen() * qpresults.mu_in_inv);
		qpwork.ruiz.scale_primal_residual_in_place_in(
				VectorViewMut<T>{from_eigen, qpwork.primal_residual_in_scaled_up});
		qpwork.ruiz.scale_primal_residual_in_place_in(
				VectorViewMut<T>{from_eigen, qpwork.primal_residual_in_scaled_low});
		qpwork.ruiz.scale_dual_in_place_in(ze);
	}

	isize numactive_inequalities = qpwork.active_inequalities.count();
	isize inner_pb_dim = qpmodel.dim + qpmodel.n_eq + numactive_inequalities;
	qpwork.rhs.setZero();
	if (qpmodel.n_in != 0) {
		qpwork.active_part_z.setZero();
		qp::line_search::active_set_change(qpmodel, qpresults, qpwork);
	}

	qpwork.rhs.head(qpmodel.dim) = -qpwork.dual_residual_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) =
			-qpwork.primal_residual_eq_scaled;
	for (isize i = 0; i < qpmodel.n_in; i++) {
		isize j = qpwork.current_bijection_map(i);
		if (j < qpresults.n_c) {
			if (qpwork.active_set_up(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_up(i);
			} else if (qpwork.active_set_low(i)) {
				qpwork.rhs(j + qpmodel.dim + qpmodel.n_eq) =
						-qpwork.primal_residual_in_scaled_low(i);
			}
		} else {
			qpwork.rhs.head(qpmodel.dim).noalias() +=
					qpresults.z(i) *
					qpwork.C_scaled.row(i); // unactive unrelevant columns
		}
	}
	iterative_solve_with_permut_fact( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork,
			eps_int,
			inner_pb_dim);
	// use active_part_z as a temporary variable to permut back dw_aug newton step
	for (isize j = 0; j < qpmodel.n_in; ++j) {
		isize i = qpwork.current_bijection_map(j);
		if (i < qpresults.n_c) {
			qpwork.active_part_z(j) = qpwork.dw_aug(qpmodel.dim + qpmodel.n_eq + i);
		} else {
			qpwork.active_part_z(j) = -qpresults.z(j);
		}
	}
	if (qpmodel.n_in != 0) {
		qpwork.dw_aug.tail(qpmodel.n_in) = qpwork.active_part_z;
		qpwork.primal_residual_in_scaled_up.noalias() +=
				(ze.to_eigen() * qpresults.mu_in_inv);
		qpwork.primal_residual_in_scaled_low.noalias() +=
				(ze.to_eigen() * qpresults.mu_in_inv);
		qpwork.Cdx.noalias() = qpwork.C_scaled * qpwork.dw_aug.head(qpmodel.dim);
		qpwork.dual_residual_scaled.noalias() -=
				qpwork.C_scaled.transpose() *
				ze.to_eigen(); // contains now Hx_prev + g + Aty_prev
	}

	qpwork.Adx =
			-qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq) * qpresults.mu_eq_inv;
	qpwork.Adx.noalias() += qpwork.A_scaled * qpwork.dw_aug.head(qpmodel.dim);

	qpwork.Hdx.noalias() = qpwork.H_scaled * qpwork.dw_aug.head(qpmodel.dim) +
	                       qpwork.A_scaled.transpose() *
	                           qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);
	qpwork.Hdx.noalias() += qpresults.rho * qpwork.dw_aug.head(qpmodel.dim);

	if (qpmodel.n_in != 0) {
		qp::line_search::initial_guess_ls(qpsettings, qpmodel, qpresults, qpwork);

		if (qpsettings.verbose) {
			std::cout << "alpha from initial guess " << qpwork.alpha << std::endl;
		}

		qpwork.primal_residual_in_scaled_up += qpwork.alpha * qpwork.Cdx;
		qpwork.primal_residual_in_scaled_low += qpwork.alpha * qpwork.Cdx;
		qpwork.active_set_up.array() =
				(qpwork.primal_residual_in_scaled_up.array() >= 0.);
		qpwork.active_set_low.array() =
				(qpwork.primal_residual_in_scaled_low.array() <= 0.);
		qpwork.active_inequalities.noalias() =
				qpwork.active_set_up || qpwork.active_set_low;
		qpwork.active_part_z.noalias() =
				qpresults.z + qpwork.alpha * qpwork.dw_aug.tail(qpmodel.n_in);
		qpwork.primal_residual_in_scaled_up_plus_alphaCdx.noalias() =
				(qpwork.active_part_z.array() > T(0.))
						.select(
								qpwork.active_part_z,
								Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));
		qpwork.primal_residual_in_scaled_low_plus_alphaCdx.noalias() =
				(qpwork.active_part_z.array() < T(0.))
						.select(
								qpwork.active_part_z,
								Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));
		qpresults.z.noalias() =
				(qpwork.active_set_up)
						.select(
								qpwork.primal_residual_in_scaled_up_plus_alphaCdx,
								Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in)) +
				(qpwork.active_set_low)
						.select(
								qpwork.primal_residual_in_scaled_low_plus_alphaCdx,
								Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in)) +
				(!qpwork.active_inequalities.array())
						.select(
								qpwork.active_part_z,
								Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(qpmodel.n_in));
	}

	qpresults.x.noalias() += (qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim));
	qpresults.y.noalias() +=
			(qpwork.alpha * qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq));

	qpwork.primal_residual_eq_scaled.noalias() +=
			(qpwork.alpha * qpwork.Adx); // contains now Ax-b - (y-y_prev)/mu
	qpwork.dual_residual_scaled.noalias() +=
			qpwork.alpha * (qpwork.Hdx); // contains now Hx + rho(x-xprev) + g + Aty
	qpwork.dw_aug.setZero();

	T err_saddle_point = compute_primal_dual_residual(qpmodel, qpresults, qpwork);
	if (std::abs(qpwork.alpha) < 1.E-10) { // TODO(check)
		err_saddle_point = 1.;
	}
	return err_saddle_point;
}

template <typename T>
T correction_guess(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps_int) {

	T err_in = 1.e6;

	for (i64 iter = 0; iter <= qpsettings.max_iter_in; ++iter) {

		if (iter == qpsettings.max_iter_in) {
			qpresults.n_tot += qpsettings.max_iter_in;
			break;
		}

		qp::detail::newton_step<T>(qpsettings, qpmodel, qpresults, qpwork, eps_int);

		qpwork.Hdx.noalias() = qpwork.H_scaled * qpwork.dw_aug.head(qpmodel.dim);
		qpwork.Adx.noalias() =
				qpwork.A_scaled *
				qpwork.dw_aug.head(
						qpmodel.dim); // try replacing it by qpresults.mu_eq *
		                      // qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq);
		qpwork.Cdx.noalias() = qpwork.C_scaled * qpwork.dw_aug.head(qpmodel.dim);

		if (qpmodel.n_in > 0) {
			qp::line_search::correction_guess_ls(qpmodel, qpresults, qpwork);
		}

		if (infty_norm(qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim)) < 1.E-11 &&
		    iter > 0) {
			qpresults.n_tot += iter + 1;
			if (qpsettings.verbose) {
				std::cout << "infty_norm(alpha_step * dx) "
									<< infty_norm(qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim))
									<< std::endl;
			}
			break;
		}

		qpresults.x.noalias() += (qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim));
		qpwork.primal_residual_in_scaled_up.noalias() +=
				(qpwork.alpha * qpwork.Cdx);
		qpwork.primal_residual_in_scaled_low.noalias() +=
				(qpwork.alpha * qpwork.Cdx);
		qpwork.primal_residual_eq_scaled.noalias() += qpwork.alpha * qpwork.Adx;
		qpresults.y.noalias() = qpresults.mu_eq * qpwork.primal_residual_eq_scaled;
		qpresults.z.noalias() =
				(qp::detail::positive_part(qpwork.primal_residual_in_scaled_up) +
		     qp::detail::negative_part(qpwork.primal_residual_in_scaled_low)) *
				qpresults.mu_in;
		qpwork.dual_residual_scaled.noalias() = qpwork.H_scaled * qpresults.x;
		T rhs_c = max2(
				qpwork.correction_guess_rhs_g, infty_norm(qpwork.dual_residual_scaled));
		qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * (qpresults.y);
		qpwork.dual_residual_scaled.noalias() += qpwork.CTz;
		rhs_c = max2(rhs_c, infty_norm(qpwork.CTz));
		qpwork.CTz.noalias() = qpwork.C_scaled.transpose() * (qpresults.z);
		qpwork.dual_residual_scaled.noalias() += qpwork.CTz;
		rhs_c = max2(rhs_c, infty_norm(qpwork.CTz));
		qpwork.dual_residual_scaled.noalias() +=
				qpwork.g_scaled + qpresults.rho * (qpresults.x - qpwork.x_prev);
		rhs_c += 1.;

		err_in = infty_norm(qpwork.dual_residual_scaled);
		if (qpsettings.verbose) {
			std::cout << "---it in " << iter << " projection norm " << err_in
								<< " alpha " << qpwork.alpha << " rhs " << eps_int * rhs_c
								<< std::endl;
		}

		if (err_in <= eps_int * rhs_c) {
			qpresults.n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}

template <typename T>
T primal_dual_newton_semi_smooth(
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork,
		T eps_int) {

	/* MUST CONTAIN IN ENTRY WITH x = x_prev ; y = y_prev ; z = z_prev
	 *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
	 *  primal_residual_eq_scaled = Ax-b+1./mu_eq (y_prev)
	 *  primal_residual_in_scaled_up = Cx-u+1./mu_in(z_prev)
	 *  primal_residual_in_scaled_low = Cx-l+1./mu_in(z_prev)
	 */

	T err_in = 1.e6;

	for (i64 iter = 0; iter <= qpsettings.max_iter_in; ++iter) {

		if (iter == qpsettings.max_iter_in) {
			qpresults.n_tot += qpsettings.max_iter_in;
			break;
		}

		qp::detail::primal_dual_semi_smooth_newton_step<T>(
				qpsettings, qpmodel, qpresults, qpwork, eps_int);

		qpwork.Hdx.noalias() = qpwork.H_scaled * qpwork.dw_aug.head(qpmodel.dim);
		qpwork.Adx.noalias() =
				qpwork.A_scaled *
				qpwork.dw_aug.head(
						qpmodel.dim); // try replacing it by qpresults.mu_eq *
		                      // qpwork.dw_aug.segment(qpmodel.dim,qpmodel.n_eq);
		qpwork.Cdx.noalias() = qpwork.C_scaled * qpwork.dw_aug.head(qpmodel.dim);

		if (qpmodel.n_in > 0) {
			qp::line_search::primal_dual_ls(qpmodel, qpresults, qpwork);
		}

		if (infty_norm(qpwork.alpha * qpwork.dw_aug) < 1.E-11 && iter > 0) {
			qpresults.n_tot += iter + 1;
			if (qpsettings.verbose) {
				std::cout << "infty_norm(alpha_step * dx) "
									<< infty_norm(qpwork.alpha * qpwork.dw_aug) << std::endl;
			}
			break;
		}

		qpresults.x.noalias() += (qpwork.alpha * qpwork.dw_aug.head(qpmodel.dim));
		qpwork.primal_residual_in_scaled_up.noalias() +=
				(qpwork.alpha *
		     qpwork.Cdx); // contains now :  C(x+alpha dx)-u + z_prev/mu_in
		qpwork.primal_residual_in_scaled_low.noalias() +=
				(qpwork.alpha *
		     qpwork.Cdx); // contains now :  C(x+alpha dx)-l + z_prev/mu_in
		qpwork.primal_residual_eq_scaled.noalias() +=
				qpwork.alpha *
				qpwork.Adx; // contains now :  A(x+alpha dx)-b + y_prev/mu_in
		qpresults.y.noalias() +=
				(qpwork.alpha * qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq));
		qpresults.z.noalias() += (qpwork.alpha * qpwork.dw_aug.tail(qpmodel.n_in));

		qpwork.dual_residual_scaled.noalias() = qpwork.H_scaled * qpresults.x;

		// T rhs_c = max2(
		//		qpwork.correction_guess_rhs_g,
		// infty_norm(qpwork.dual_residual_scaled));

		qpwork.dual_residual_scaled.noalias() +=
				qpwork.g_scaled + qpresults.rho * (qpresults.x - qpwork.x_prev);

		qpwork.CTz.noalias() = qpwork.A_scaled.transpose() * (qpresults.y);
		qpwork.dual_residual_scaled.noalias() += qpwork.CTz;
		// rhs_c = max2(rhs_c, infty_norm(qpwork.CTz));

		qpwork.CTz.noalias() = qpwork.C_scaled.transpose() * (qpresults.z);
		qpwork.dual_residual_scaled.noalias() +=
				qpwork.CTz; // contains now : Hx + rho (x-x_prev) + A.Ty + C.T z
		// rhs_c = max2(rhs_c, infty_norm(qpwork.CTz));
		/*
		qpwork.err.segment(qpmodel.dim,qpmodel.n_eq).noalias() =
		qpwork.primal_residual_eq_scaled + qpwork.b_scaled ; // A(x+alpha dx) +
		y_prev/mu_in rhs_c = max2(rhs_c,
		infty_norm(qpwork.err.segment(qpmodel.dim,qpmodel.n_eq))); rhs_c =
		max2(rhs_c, qpwork.correction_guess_rhs_b ) ;

		qpwork.err.tail(qpmodel.n_in).noalias() =
		qpwork.primal_residual_in_scaled_up + qpwork.u_scaled ; // C(x+alpha dx) +
		z_prev/mu_in rhs_c = max2(rhs_c, infty_norm(qpwork.err.tail(qpmodel.n_in)));
		rhs_c += 1.;
		*/
		err_in = compute_inner_loop_saddle_point(qpmodel, qpresults, qpwork);

		if (qpsettings.verbose) {
			std::cout << "---it in " << iter << " projection norm " << err_in
								<< " alpha " << qpwork.alpha << std::endl;
		}

		if (err_in <= eps_int) {
			qpresults.n_tot += iter + 1;
			break;
		}
	}

	return err_in;
}

template <typename T>
void qp_solve( //
		const qp::QPSettings<T>& qpsettings,
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	using namespace ldlt::tags;

	/*** TEST WITH MATRIX FULL OF NAN FOR DEBUG
	  static constexpr Layout layout = rowmajor;
	  static constexpr auto DYN = Eigen::Dynamic;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	RowMat test(2,2); // test it is full of nan for debug
	std::cout << "test " << test << std::endl;
	*/

	//::Eigen::internal::set_is_malloc_allowed(false);

	const T machine_eps = std::numeric_limits<T>::epsilon();

	T bcl_eta_ext_init = pow(T(0.1), qpsettings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(qpsettings.eps_abs, T(1.E-9));

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);
	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	isize saturation(0);

	for (i64 iter = 0; iter <= qpsettings.max_iter; ++iter) {

		qpresults.n_ext += 1;
		if (iter == qpsettings.max_iter) {
			break;
		}

		// compute primal residual

		qp::detail::global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		qp::detail::global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);

		T new_bcl_mu_in(qpresults.mu_in);
		T new_bcl_mu_eq(qpresults.mu_eq);
		T new_bcl_mu_in_inv(qpresults.mu_in_inv);
		T new_bcl_mu_eq_inv(qpresults.mu_eq_inv);

		T rhs_pri(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_pri +=
					qpsettings.eps_rel *
					max2(
							max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
							max2(
									max2(
											qpwork.primal_feasibility_rhs_1_eq,
											qpwork.primal_feasibility_rhs_1_in_u),
									qpwork.primal_feasibility_rhs_1_in_l));
		}
		bool is_primal_feasible = primal_feasibility_lhs <= rhs_pri;

		T rhs_dua(qpsettings.eps_abs);
		if (qpsettings.eps_rel != 0) {
			rhs_dua +=
					qpsettings.eps_rel *
					max2(
							max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
							max2(dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2));
		}

		bool is_dual_feasible = dual_feasibility_lhs <= rhs_dua;

		if (qpsettings.verbose) {
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << "bcl_eta_ext : " << bcl_eta_ext
								<< " bcl_eta_in : " << bcl_eta_in << " rho : " << qpresults.rho
								<< " bcl_mu_eq : " << qpresults.mu_eq
								<< " bcl_mu_in : " << qpresults.mu_in << std::endl;
			std::cout << "qpsettings.eps_abs " << qpsettings.eps_abs
								<< "  qpsettings.eps_rel *rhs "
								<< qpsettings.eps_rel *
											 max2(
													 max2(
															 primal_feasibility_eq_rhs_0,
															 primal_feasibility_in_rhs_0),
													 max2(
															 max2(
																	 qpwork.primal_feasibility_rhs_1_eq,
																	 qpwork.primal_feasibility_rhs_1_in_u),
															 qpwork.primal_feasibility_rhs_1_in_l))
								<< std::endl;
			std::cout << "is_primal_feasible " << is_primal_feasible
								<< " is_dual_feasible " << is_dual_feasible << std::endl;
		}
		if (is_primal_feasible) {

			if (dual_feasibility_lhs >=
			        qpsettings.refactor_dual_feasibility_threshold &&
			    qpresults.rho != qpsettings.refactor_rho_threshold) {

				T rho_new(qpsettings.refactor_rho_threshold);

				refactorize(qpmodel, qpresults, qpwork, rho_new);

				qpresults.rho = rho_new;
			}
			if (is_dual_feasible) {

				qpwork.ruiz.unscale_primal_in_place(
						VectorViewMut<T>{from_eigen, qpresults.x});
				qpwork.ruiz.unscale_dual_in_place_eq(
						VectorViewMut<T>{from_eigen, qpresults.y});
				qpwork.ruiz.unscale_dual_in_place_in(
						VectorViewMut<T>{from_eigen, qpresults.z});

				{
					// EigenAllowAlloc _{};
					// T result = 0;
					for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
						qpresults.objValue +=
								0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
						qpresults.objValue +=
								qpresults.x(j) *
								T(qpmodel.H.col(j)
						          .tail(qpmodel.dim - j - 1)
						          .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
					}
					qpresults.objValue += (qpmodel.g).dot(qpresults.x);
					// qpresults.objValue =
					//		(0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x);
				}
				break;
			}
		}

		qpwork.x_prev = qpresults.x;
		qpwork.y_prev = qpresults.y;
		qpwork.z_prev = qpresults.z;

		// primal dual version from gill and robinson

		// dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z as x =
		// x_prev
		// std::cout << " intermediate algebra PDAL no IG " << std::endl;
		qpwork.primal_residual_eq_scaled.noalias() +=
				(qpwork.y_prev *
		     qpresults
		         .mu_eq_inv); // contains now Ax-b+1./mu_eq * y_prev as y = y_prev
		qpwork.ruiz.scale_primal_residual_in_place_in(VectorViewMut<T>{
				from_eigen,
				qpwork.primal_residual_in_scaled_up}); // contains now scaled(Cx)
		qpwork.primal_residual_in_scaled_up.noalias() +=
				qpwork.z_prev *
				qpresults.mu_in_inv; // contains now scaled(Cx+z_prev/mu_in)
		qpwork.primal_residual_in_scaled_low = qpwork.primal_residual_in_scaled_up;
		qpwork.primal_residual_in_scaled_up -=
				qpwork.u_scaled; // contains now scaled(Cx-u+z_prev/mu_in)
		qpwork.primal_residual_in_scaled_low -=
				qpwork.l_scaled; // contains now scaled(Cx-l+z_prev/mu_in)

		T err_in = qp::detail::primal_dual_newton_semi_smooth(
				qpsettings, qpmodel, qpresults, qpwork, bcl_eta_in);
		if (qpsettings.verbose) {
			std::cout << " error from correction guess : " << err_in << std::endl;
		}

		T primal_feasibility_lhs_new(primal_feasibility_lhs);

		qp::detail::global_primal_residual(
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_eq_lhs,
				primal_feasibility_in_lhs);

		is_primal_feasible =
				primal_feasibility_lhs_new <=
				(qpsettings.eps_abs +
		     qpsettings.eps_rel *
		         max2(
								 max2(primal_feasibility_eq_rhs_0, primal_feasibility_in_rhs_0),
								 max2(
										 max2(
												 qpwork.primal_feasibility_rhs_1_eq,
												 qpwork.primal_feasibility_rhs_1_in_u),
										 qpwork.primal_feasibility_rhs_1_in_l)));

		if (is_primal_feasible) {
			T dual_feasibility_lhs_new(dual_feasibility_lhs);

			qp::detail::global_dual_residual(
					qpmodel,
					qpresults,
					qpwork,
					dual_feasibility_lhs_new,
					dual_feasibility_rhs_0,
					dual_feasibility_rhs_1,
					dual_feasibility_rhs_3);

			is_dual_feasible =
					dual_feasibility_lhs_new <=
					(qpsettings.eps_abs +
			     qpsettings.eps_rel *
			         max2(
									 max2(dual_feasibility_rhs_3, dual_feasibility_rhs_0),
									 max2(
											 dual_feasibility_rhs_1, qpwork.dual_feasibility_rhs_2)));

			if (is_dual_feasible) {

				qpwork.ruiz.unscale_primal_in_place(
						VectorViewMut<T>{from_eigen, qpresults.x});
				qpwork.ruiz.unscale_dual_in_place_eq(
						VectorViewMut<T>{from_eigen, qpresults.y});
				qpwork.ruiz.unscale_dual_in_place_in(
						VectorViewMut<T>{from_eigen, qpresults.z});

				{
					// EigenAllowAlloc _{};
					// qpresults.objValue =
					//		(0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x);
					for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
						qpresults.objValue +=
								0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
						qpresults.objValue +=
								qpresults.x(j) *
								T(qpmodel.H.col(j)
						          .tail(qpmodel.dim - j - 1)
						          .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
					}
					qpresults.objValue += (qpmodel.g).dot(qpresults.x);
				}
				break;
			}
		}

		qp::detail::bcl_update(
				qpsettings,
				qpmodel,
				qpresults,
				qpwork,
				primal_feasibility_lhs_new,
				bcl_eta_ext,
				bcl_eta_in,
				bcl_eta_ext_init,
				eps_in_min,

				new_bcl_mu_in,
				new_bcl_mu_eq,
				new_bcl_mu_in_inv,
				new_bcl_mu_eq_inv

		);

		// COLD RESTART

		T dual_feasibility_lhs_new(dual_feasibility_lhs);

		qp::detail::global_dual_residual(
				qpmodel,
				qpresults,
				qpwork,
				dual_feasibility_lhs_new,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_3);

		if ((primal_feasibility_lhs_new /
		         max2(primal_feasibility_lhs, machine_eps) >=
		     1.) &&
		    (dual_feasibility_lhs_new / max2(primal_feasibility_lhs, machine_eps) >=
		     1.) &&
		    qpresults.mu_in >= 1.E5) {

			if (qpsettings.verbose) {
				std::cout << "cold restart" << std::endl;
			}

			new_bcl_mu_in = qpsettings.cold_reset_mu_in;
			new_bcl_mu_eq = qpsettings.cold_reset_mu_eq;
			new_bcl_mu_in_inv = qpsettings.cold_reset_mu_in_inv;
			new_bcl_mu_eq_inv = qpsettings.cold_reset_mu_eq_inv;
		}

		/// effective mu upddate

		if (qpresults.mu_in != new_bcl_mu_in || qpresults.mu_eq != new_bcl_mu_eq) {
			{ ++qpresults.n_mu_change; }
			qp::detail::mu_update(
					qpmodel, qpresults, qpwork, new_bcl_mu_eq_inv, new_bcl_mu_in_inv);
		}

		qpresults.mu_eq = new_bcl_mu_eq;
		qpresults.mu_in = new_bcl_mu_in;
		qpresults.mu_eq_inv = new_bcl_mu_eq_inv;
		qpresults.mu_in_inv = new_bcl_mu_in_inv;
	}

	{
		// EigenAllowAlloc _{};
		// qpresults.objValue =
		//		(0.5 * qpmodel.H * qpresults.x + qpmodel.g).dot(qpresults.x);
		for (Eigen::Index j = 0; j < qpmodel.dim; ++j) {
			qpresults.objValue +=
					0.5 * (qpresults.x(j) * qpresults.x(j)) * qpmodel.H(j, j);
			qpresults.objValue +=
					qpresults.x(j) * T(qpmodel.H.col(j)
			                           .tail(qpmodel.dim - j - 1)
			                           .dot(qpresults.x.tail(qpmodel.dim - j - 1)));
		}
		qpresults.objValue += (qpmodel.g).dot(qpresults.x);
	}
}

template <typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using MatRef =
		Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const>;

template <typename Mat, typename T>
void QPsetup_generic( //
		Mat const& H,
		VecRef<T> g,
		Mat const& A,
		VecRef<T> b,
		Mat const& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,

		T eps_abs = 1.e-9,
		T eps_rel = 0,
		const bool VERBOSE = true,
		const bool PMM = true

) {

	QPSettings.eps_abs = eps_abs;
	QPSettings.eps_rel = eps_rel;
	QPSettings.verbose = VERBOSE;
	if (PMM) {
		QPSettings.solvingMethod = InnerLoopSolvingMethod::pmm;
	} else {
		QPSettings.solvingMethod = InnerLoopSolvingMethod::pdal;
	}

	qpmodel.H = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
					H.eval());
	// qpmodel.H = Eigen::MatrixXd(H);
	qpmodel.g = g.eval();
	qpmodel.A = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
					A.eval());
	// qpmodel.A = Eigen::MatrixXd(A);
	qpmodel.b = b.eval();
	// qpmodel.C = Eigen::MatrixXd(C);
	qpmodel.C = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
					C.eval());
	qpmodel.u = u.eval();
	qpmodel.l = l.eval();

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	qpwork.ruiz.scale_qp_in_place(
			qp_scaled, VectorViewMut<T>{from_eigen, qpwork.dw_aug});
	qpwork.dw_aug.setZero();

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);
	qpwork.correction_guess_rhs_b = infty_norm(qpwork.b_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv);

	{
		LDLT_MAKE_STACK(stack, ldlt::Ldlt<T>::factor_req(qpwork.kkt.rows()));
		qpwork.ldl.factor(qpwork.kkt, LDLT_FWD(stack));
	}

	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	// std::cout << "-qpwork.g_scaled " << -qpwork.g_scaled << std::endl;
	// std::cout << "qpwork.b_scaled " << qpwork.b_scaled << std::endl;

	qp::detail::iterative_solve_with_permut_fact( //
			QPSettings,
			qpmodel,
			QPResults,
			qpwork,
			T(1),
			qpmodel.dim + qpmodel.n_eq);

	QPResults.x = qpwork.dw_aug.head(qpmodel.dim);
	QPResults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_htot,
	     Uninit,
	     Mat(qpmodel.dim + qpmodel.n_eq, qpmodel.dim + qpmodel.n_eq),
	     LDLT_CACHELINE_BYTES,
	     T));
	auto Htot = _htot.to_eigen().eval();
	Htot.setZero();
	Htot.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	Htot.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	Htot.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	Htot.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	Htot.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	Htot.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv);

	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;
	// std::cout << " norm(Htot * dw -  rhs) " << infty_norm(Htot *
	// qpwork.dw_aug.head(qpmodel.dim+qpmodel.n_eq) -
	// qpwork.rhs.head(qpmodel.dim+qpmodel.n_eq)) << std::endl;

	// std::cout<< "Htot " << Htot << std::endl;

	// std::cout << "x ws " << QPResults.x << std::endl;
	// std::cout << "y ws " << QPResults.y << std::endl;

	qpwork.dw_aug.setZero();
	qpwork.rhs.setZero();
}

template <typename T>
void QPsetup_dense( //
		MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,

		T eps_abs = 1.e-9,
		T eps_rel = 0,
		const bool VERBOSE = true,
		const bool PMM = true) {
	detail::QPsetup_generic(
			H,
			g,
			A,
			b,
			C,
			u,
			l,
			QPSettings,
			qpmodel,
			qpwork,
			QPResults,
			eps_abs,
			eps_rel,
			VERBOSE,
			PMM);
}

template <typename T>
void QPsetup( //
		const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l,
		qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,

		T eps_abs = 1.e-9,
		T eps_rel = 0,
		const bool VERBOSE = true,
		const bool PMM = true

) {
	detail::QPsetup_generic(
			H,
			g,
			A,
			b,
			C,
			u,
			l,
			QPSettings,
			qpmodel,
			qpwork,
			QPResults,
			eps_abs,
			eps_rel,
			VERBOSE,
			PMM);
}

} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_SOLVER_HPP_HDWGZKCLS */
