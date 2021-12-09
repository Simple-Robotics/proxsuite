#ifndef INRIA_LDLT_Qpsetup_HPP_VCVSK3EOS
#define INRIA_LDLT_Qpsetup_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include <qp/QPWorkspace.hpp>
#include <qp/QPData.hpp>
#include <qp/QPResults.hpp>
#include <qp/QPSettings.hpp>

namespace qp {

template <typename T>
struct Qpsetup {
public:

    static constexpr auto DYN = Eigen::Dynamic;

    enum { layout = Eigen::RowMajor };

	using ColMat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    qp::Qpsettings<T> _qpsettings;
    qp::Qpresults<T> _qpresults;
    qp::Qpdata<T> _qpmodel;
	qp::Qpworkspace<T> _qpwork;

	Qpsetup(ColMat H,Vec g, ColMat A, Vec b, ColMat C, Vec u, Vec l,
            isize n_max, isize n_max_in,
            T mu_max_eq=1.e6,T mu_max_in=1.e6,
            T mu_f=10,bool VERBOSE=false){

                    isize dim = H.rows();
                    isize n_eq = A.rows();
                    isize n_in = C.rows();

                    qp::Qpresults<T> qpresults{dim,n_eq,n_in};
                    qp::Qpsettings<T> qpsettings{};
                    qpsettings._mu_update_factor = mu_f;
                    qpsettings._mu_update_inv_factor = 1./mu_f;
                    qpsettings._VERBOSE = VERBOSE;
                    qpsettings._max_iter = n_max;
                    qpsettings._max_iter_in = n_max_in;
                    qpsettings._mu_max_eq = mu_max_eq;
                    qpsettings._mu_max_eq_inv = T(1)/mu_max_eq;
                    qpsettings._mu_max_in = mu_max_in;
                    qpsettings._mu_max_in_inv = T(1)/mu_max_in;

                    qp::Qpdata<T> qpmodel{H,g,A,b,C,u,l};
                    qp::Qpworkspace<T> qpwork{dim, n_eq, n_in};
                    
                    qpwork._h_scaled = qpmodel._H;
                    qpwork._g_scaled = qpmodel._g;
                    qpwork._a_scaled = qpmodel._A;
                    qpwork._b_scaled = qpmodel._b;
                    qpwork._c_scaled = qpmodel._C;
                    qpwork._u_scaled = qpmodel._u;
                    qpwork._l_scaled = qpmodel._l;

                    auto qp_scaled = qp::QpViewBoxMut<T>{
                            MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._h_scaled},
                            VectorViewMut<T>{ldlt::from_eigen, qpwork._g_scaled},
                            MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._a_scaled},
                            VectorViewMut<T>{ldlt::from_eigen, qpwork._b_scaled},
                            MatrixViewMut<T,rowmajor>{ldlt::from_eigen, qpwork._c_scaled},
                            VectorViewMut<T>{ldlt::from_eigen, qpwork._u_scaled},
                            VectorViewMut<T>{ldlt::from_eigen, qpwork._l_scaled}};

                    qpwork._primal_feasibility_rhs_1_eq = infty_norm(qpmodel._b);
                    qpwork._primal_feasibility_rhs_1_in_u = infty_norm(qpmodel._u);
                    qpwork._primal_feasibility_rhs_1_in_l = infty_norm(qpmodel._l);
                    qpwork._dual_feasibility_rhs_2 = infty_norm(qpmodel._g);

                    qpwork._ruiz.scale_qp_in_place(qp_scaled,VectorViewMut<T>{ldlt::from_eigen,qpwork._dw_aug}); // avoids temporary allocation in ruiz using another unused for the moment preallocated variable in qpwork
                    
                    qpwork._correction_guess_rhs_g = infty_norm(qpwork._g_scaled);

                    qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim) = qp_scaled.H.to_eigen();
                    qpwork._kkt.topLeftCorner(qpmodel._dim, qpmodel._dim).diagonal().array() += qpresults._rho;	
                    qpwork._kkt.block(0, qpmodel._dim, qpmodel._dim, qpmodel._n_eq) = qp_scaled.A.to_eigen().transpose();
                    qpwork._kkt.block(qpmodel._dim, 0, qpmodel._n_eq, qpmodel._dim) = qp_scaled.A.to_eigen();
                    qpwork._kkt.bottomRightCorner(qpmodel._n_eq, qpmodel._n_eq).setZero();
                    qpwork._kkt.diagonal().segment(qpmodel._dim, qpmodel._n_eq).setConstant(-qpresults._mu_eq_inv); // mu stores the inverse of mu

                    _qpsettings = qpsettings;
                    _qpwork = qpwork;
                    _qpmodel = qpmodel;
                    _qpresults = qpresults;
                    _qpsettings = qpsettings;

                }
    
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_Qpsetup_HPP_VCVSK3EOS */
