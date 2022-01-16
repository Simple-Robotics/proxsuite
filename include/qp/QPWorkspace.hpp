#ifndef INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS
#define INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include <qp/precond/ruiz.hpp>

namespace qp {

template <typename T>
struct Qpworkspace {
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

    Vec _x_prev;
    Vec _y_prev; 
    Vec _z_prev; 

    ///// KKT system storage
    ColMat _kkt;

    //// Active set & permutation vector 
    VecISize _current_bijection_map;
    VecISize _new_bijection_map;

    VecBool _active_set_up; 
    VecBool _active_set_low; 
    VecBool _active_inequalities;

    //// First order residuals for line search
    
	Vec _Hdx; 
	Vec _Cdx;
	Vec _Adx; 

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

    Vec _dual_residual_scaled;
    Vec _primal_residual_eq_scaled;
    Vec _primal_residual_in_scaled_up; 
    Vec _primal_residual_in_scaled_low; 

	Vec _primal_residual_in_scaled_up_plus_alphaCdx; 
	Vec _primal_residual_in_scaled_low_plus_alphaCdx; 
	Vec _CTz;

	Qpworkspace( isize dim=0, isize n_eq=0, isize n_in=0)
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
                _x_prev(dim),
                _y_prev(n_eq),
                _z_prev(n_in),
                _kkt(dim+n_eq,dim+n_eq),
                _current_bijection_map(n_in),
                _new_bijection_map(n_in),
                _active_set_up(n_in),
                _active_set_low(n_in),
                _active_inequalities(n_in),
                _Hdx(dim),
                _Cdx(n_in),
                _Adx(n_eq),
                _active_part_z(n_in),
                _dw_aug(dim+n_eq+n_in),
                _rhs(dim+n_eq+n_in),
                _err(dim+n_eq+n_in),

                _dual_residual_scaled(dim),
                _primal_residual_eq_scaled(n_eq),
                _primal_residual_in_scaled_up(n_in),
                _primal_residual_in_scaled_low(n_in),

                _primal_residual_in_scaled_up_plus_alphaCdx(n_in),
                _primal_residual_in_scaled_low_plus_alphaCdx(n_in),
                _CTz(dim)

            {
                    _alphas.reserve( 3*n_in );
                    _h_scaled.setZero();
                    _g_scaled.setZero();
                    _a_scaled.setZero();
                    _c_scaled.setZero();
                    _b_scaled.setZero();
                    _u_scaled.setZero();
                    _l_scaled.setZero();
                    _x_prev.setZero();
                    _y_prev.setZero();
                    _z_prev.setZero();
                    _kkt.setZero();
                    for (isize i = 0; i < n_in; i++) {
                        _current_bijection_map(i) = i;
                        _new_bijection_map(i) = i;
                    }
                    _Hdx.setZero();
                    _Cdx.setZero();
                    _Adx.setZero();
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

                    _dual_residual_scaled.setZero();
                    _primal_residual_eq_scaled.setZero();
                    _primal_residual_in_scaled_up.setZero();
                    _primal_residual_in_scaled_low.setZero();

                    _primal_residual_in_scaled_up_plus_alphaCdx.setZero();
                    _primal_residual_in_scaled_low_plus_alphaCdx.setZero();
                    _CTz.setZero();
            }

        
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_OLD_NEW_QPWORKSPACE_HPP_VCVSK3EOS */
