#ifndef INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS
#define INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include <qp/precond/ruiz.hpp>

namespace qp {

template <typename T>
struct OldNew_Qpworkspace {
public:
	static constexpr auto DYN = Eigen::Dynamic;
    enum { layout = Eigen::RowMajor };

	using ColMat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	using VecMap = Eigen::Map<Vec const>;
	using VecMapMut = Eigen::Map<Vec>;

    using MatMap = Eigen::Map<ColMat const>;
	using MatMapMut = Eigen::Map<ColMat>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
    using VecISize = Eigen::Matrix<isize, DYN, 1>;

    using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
    using VecBool = Eigen::Matrix<bool, DYN, 1>;

    ///// Equilibrator
    qp::preconditioner::RuizEquilibration<T> _ruiz;

    ///// Cholesky Factorization
    ldlt::Ldlt<T> _ldl;

    ///// QP STORAGE
    ColMat _h_scaled;
    Vec _g_scaled;
    ColMat _a_scaled;
    ColMat _c_scaled;
    Vec _b_scaled;
    Vec _u_scaled;
    Vec _l_scaled;

    ///// Initial variable loading

    Vec _xe;
    Vec _ye;
    Vec _ze;

    ///// KKT system storage
    ColMat _kkt;

    //// Active set & permutation vector 
    VecISize _current_bijection_map;
    VecISize _new_bijection_map;

    VecBool _l_active_set_n_u;
    VecBool _l_active_set_n_l;
    VecBool _active_inequalities;

    //// First order residuals for line search
    
	Vec _d_dual_for_eq;
	Vec _Cdx;
	Vec _d_primal_residual_eq;

    Vec _active_part_z;
    std::vector<T> _alphas;

    ///// Newton variables
    Vec _dw_aug;
    Vec _rhs;
    Vec _err;
    
    //// Relative residuals constants

    T _primal_feasibility_rhs_1_eq;
    T _primal_feasibility_rhs_1_in_u;
    T _primal_feasibility_rhs_1_in_l;
    T _dual_feasibility_rhs_2;
    T _correction_guess_rhs_g;
    T _alpha;

    ///// Supplementary auxiliary variable : TODO : optimize by reducing their number

	Vec _residual_scaled;
	Vec _residual_scaled_tmp;
	Vec _residual_unscaled;
	Vec _tmp_u;
	Vec _tmp_l;
	Vec _aux_u;
	Vec _aux_l;
	Vec _dz_p;
	Vec _tmp1;
	Vec _tmp2;
	Vec _tmp3;
	Vec _tmp4;
	Vec _tmp_d2_u;
	Vec _tmp_d2_l;
	Vec _tmp_d3;
	Vec _tmp2_u;
	Vec _tmp2_l;
	Vec _tmp3_local_saddle_point;
	VecISize _active_set_l;
	VecISize _active_set_u;
	VecISize _inactive_set;

	OldNew_Qpworkspace( isize dim=0, isize n_eq=0, isize n_in=0)
			: //
                _ruiz(qp::preconditioner::RuizEquilibration<T>{dim,n_eq + n_in}),
                _ldl(ldlt::reserve_uninit, dim+n_eq), // old version with alloc
                _h_scaled(dim, dim),
				_g_scaled(dim),
				_a_scaled(n_eq,dim),
				_c_scaled(n_in,dim),
                _b_scaled(n_eq),
                _u_scaled(n_in),
                _l_scaled(n_in),
                _xe(dim),
                _ye(n_eq),
                _ze(n_in),
                _kkt(dim+n_eq,dim+n_eq),
                _current_bijection_map(n_in),
                _new_bijection_map(n_in),
                _l_active_set_n_u(n_in),
                _l_active_set_n_l(n_in),
                _active_inequalities(n_in),
                _d_dual_for_eq(dim),
                _Cdx(n_in),
                _d_primal_residual_eq(n_eq),
                _active_part_z(n_in),
                _dw_aug(dim+n_eq+n_in),
                _rhs(dim+n_eq+n_in),
                _err(dim+n_eq+n_in),

                _residual_scaled(dim+n_eq+n_in),
                _residual_scaled_tmp(dim+n_eq+2*n_in),
                _residual_unscaled(dim+n_eq+2*n_in),
                _tmp_u(n_in),
                _tmp_l(n_in),
                _aux_u(dim),
                _aux_l(dim),
                _dz_p(n_in),
                _tmp1(dim),
                _tmp2(dim),
                _tmp3(dim),
                _tmp4(dim),
                _tmp_d2_u(n_in),
                _tmp_d2_l(n_in),
                _tmp_d3(n_in),
                _tmp2_u(n_in),
                _tmp2_l(n_in),
                _tmp3_local_saddle_point(n_in),
                _active_set_l(n_in),
                _active_set_u(n_in),
                _inactive_set(n_in)

            {
                    _alphas.reserve( 3*n_in );
                    _h_scaled.setZero();
                    _g_scaled.setZero();
                    _a_scaled.setZero();
                    _c_scaled.setZero();
                    _b_scaled.setZero();
                    _u_scaled.setZero();
                    _l_scaled.setZero();
                    _xe.setZero();
                    _ye.setZero();
                    _ze.setZero();
                    _kkt.setZero();
                    for (isize i = 0; i < n_in; i++) {
                        _current_bijection_map(i) = i;
                        _new_bijection_map(i) = i;
                    }
                    _d_dual_for_eq.setZero();
                    _Cdx.setZero();
                    _d_primal_residual_eq.setZero();
                    _active_part_z.setZero();
                    _dw_aug.setZero();
                    _rhs.setZero();
                    _err.setZero();

                    _primal_feasibility_rhs_1_eq = 0;
                    _primal_feasibility_rhs_1_in_u = 0;
                    _primal_feasibility_rhs_1_in_l = 0;
                    _dual_feasibility_rhs_2 = 0;
                    _correction_guess_rhs_g = 0;
                    _alpha = 0.;

                    _residual_scaled.setZero();
                    _residual_scaled_tmp.setZero();
                    _residual_unscaled.setZero();
                    _tmp_u.setZero();
                    _tmp_l.setZero();
                    _aux_u.setZero();
                    _aux_l.setZero();
                    _dz_p.setZero();
                    _tmp1.setZero();
                    _tmp2.setZero();
                    _tmp3.setZero();
                    _tmp4.setZero();
                    _tmp_d2_u.setZero();
                    _tmp_d2_l.setZero();
                    _tmp_d3.setZero();
                    _tmp2_u.setZero();
                    _tmp2_l.setZero();
                    _tmp3_local_saddle_point.setZero();
                    _active_set_l.setZero();
                    _active_set_u.setZero();
                    _inactive_set.setZero();

                    
            }

        
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS */
