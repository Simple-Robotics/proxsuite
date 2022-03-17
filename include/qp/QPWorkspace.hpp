#ifndef INRIA_LDLT_OLD_NEW_QPWorkspace_HPP_VCVSK3EOS
#define INRIA_LDLT_OLD_NEW_QPWorkspace_HPP_VCVSK3EOS

#include <Eigen/Core>
#include <ldlt/ldlt.hpp>
#include <ldlt/views.hpp>
#include <dense-ldlt/ldlt.hpp>
#include <veg/vec.hpp>
#include <qp/precond/ruiz.hpp>

namespace qp {

template <typename T>
struct QPWorkspace {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	enum { layout = Eigen::RowMajor };

	using Mat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	using VecMap = Eigen::Map<Vec const>;
	using VecMapMut = Eigen::Map<Vec>;

	using MatMap = Eigen::Map<Mat const>;
	using MatMapMut = Eigen::Map<Mat>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
	using VecISize = Eigen::Matrix<isize, DYN, 1>;

	using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
	using VecBool = Eigen::Matrix<bool, DYN, 1>;

	///// Equilibrator
	qp::preconditioner::RuizEquilibration<T> ruiz;

	///// Cholesky Factorization
	dense_ldlt::Ldlt<T> ldl{};
	veg::Vec<unsigned char> ldl_stack;

	///// QP STORAGE
	Mat H_scaled;
	Vec g_scaled;
	Mat A_scaled;
	Mat C_scaled;
	Vec b_scaled;
	Vec u_scaled;
	Vec l_scaled;

	///// Initial variable loading

	Vec x_prev;
	Vec y_prev;
	Vec z_prev;

	///// KKT system storage
	Mat kkt;

	//// Active set & permutation vector
	VecISize current_bijection_map;
	VecISize new_bijection_map;

	VecBool active_set_up;
	VecBool active_set_low;
	VecBool active_inequalities;

	//// First order residuals for line search

	Vec Hdx;
	Vec Cdx;
	Vec Adx;

	Vec active_part_z;
	std::vector<T> alphas;

	///// Newton variables
	Vec dw_aug;
	Vec rhs;
	Vec err;

	//// Relative residuals constants

	T primal_feasibility_rhs_1_eq;
	T primal_feasibility_rhs_1_in_u;
	T primal_feasibility_rhs_1_in_l;
	T dual_feasibility_rhs_2;
	T correction_guess_rhs_g;
	T correction_guess_rhs_b;
	T alpha;

	Vec dual_residual_scaled;
	Vec primal_residual_eq_scaled;
	Vec primal_residual_in_scaled_up;
	Vec primal_residual_in_scaled_low;

	Vec primal_residual_in_scaled_up_plus_alphaCdx;
	Vec primal_residual_in_scaled_low_plus_alphaCdx;
	Vec CTz;

	QPWorkspace(isize dim = 0, isize n_eq = 0, isize n_in = 0)
			: //
				ruiz(qp::preconditioner::RuizEquilibration<T>{dim, n_eq + n_in}),
				ldl{}, // old version with alloc
				H_scaled(dim, dim),
				g_scaled(dim),
				A_scaled(n_eq, dim),
				C_scaled(n_in, dim),
				b_scaled(n_eq),
				u_scaled(n_in),
				l_scaled(n_in),
				x_prev(dim),
				y_prev(n_eq),
				z_prev(n_in),
				kkt(dim + n_eq, dim + n_eq),
				current_bijection_map(n_in),
				new_bijection_map(n_in),
				active_set_up(n_in),
				active_set_low(n_in),
				active_inequalities(n_in),
				Hdx(dim),
				Cdx(n_in),
				Adx(n_eq),
				active_part_z(n_in),
				dw_aug(dim + n_eq + n_in),
				rhs(dim + n_eq + n_in),
				err(dim + n_eq + n_in),

				dual_residual_scaled(dim),
				primal_residual_eq_scaled(n_eq),
				primal_residual_in_scaled_up(n_in),
				primal_residual_in_scaled_low(n_in),

				primal_residual_in_scaled_up_plus_alphaCdx(n_in),
				primal_residual_in_scaled_low_plus_alphaCdx(n_in),
				CTz(dim)

	{
		ldl.reserve_uninit(dim + n_eq + n_in);
		ldl_stack.resize_for_overwrite(
				veg::dynstack::StackReq(
						dense_ldlt::Ldlt<T>::factorize_req(dim + n_eq + n_in) |
						(dense_ldlt::temp_mat_req(veg::Tag<T>{}, dim + n_eq + n_in, n_in) &
		         dense_ldlt::Ldlt<T>::insert_block_at_req(dim + n_eq + n_in, n_in)) |
						dense_ldlt::Ldlt<T>::solve_in_place_req(dim + n_eq + n_in))
						.alloc_req());

		alphas.reserve(3 * n_in);
		H_scaled.setZero();
		g_scaled.setZero();
		A_scaled.setZero();
		C_scaled.setZero();
		b_scaled.setZero();
		u_scaled.setZero();
		l_scaled.setZero();
		x_prev.setZero();
		y_prev.setZero();
		z_prev.setZero();
		kkt.setZero();
		for (isize i = 0; i < n_in; i++) {
			current_bijection_map(i) = i;
			new_bijection_map(i) = i;
		}
		Hdx.setZero();
		Cdx.setZero();
		Adx.setZero();
		active_part_z.setZero();
		dw_aug.setZero();
		rhs.setZero();
		err.setZero();

		primal_feasibility_rhs_1_eq = 0;
		primal_feasibility_rhs_1_in_u = 0;
		primal_feasibility_rhs_1_in_l = 0;
		dual_feasibility_rhs_2 = 0;
		correction_guess_rhs_g = 0;
		correction_guess_rhs_b = 0;
		alpha = 1.;

		dual_residual_scaled.setZero();
		primal_residual_eq_scaled.setZero();
		primal_residual_in_scaled_up.setZero();
		primal_residual_in_scaled_low.setZero();

		primal_residual_in_scaled_up_plus_alphaCdx.setZero();
		primal_residual_in_scaled_low_plus_alphaCdx.setZero();
		CTz.setZero();
	}

	void reset_results(isize n_in = 0) {

		Hdx.setZero();
		Cdx.setZero();
		Adx.setZero();
		active_part_z.setZero();
		dw_aug.setZero();
		rhs.setZero();
		err.setZero();

		alpha = 1.;

		dual_residual_scaled.setZero();
		primal_residual_eq_scaled.setZero();
		primal_residual_in_scaled_up.setZero();
		primal_residual_in_scaled_low.setZero();

		primal_residual_in_scaled_up_plus_alphaCdx.setZero();
		primal_residual_in_scaled_low_plus_alphaCdx.setZero();
		CTz.setZero();

		x_prev.setZero();
		y_prev.setZero();
		z_prev.setZero();

		for (isize i = 0; i < n_in; i++) {
			current_bijection_map(i) = i;
			new_bijection_map(i) = i;
		}
	}
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_QPWorkspace_HPP_VCVSK3EOS */
