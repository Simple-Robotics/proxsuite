#ifndef INRIA_LDLT_QPData_HPP_VCVSK3EOS
#define INRIA_LDLT_QPData_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"

namespace qp {

template <typename T>
struct Qpdata {
public:
	static constexpr auto DYN = Eigen::Dynamic;
    enum { layout = Eigen::RowMajor };
	using ColMat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// QP STORAGE
    ColMat _H;
    Vec  _g;
    ColMat  _A;
    ColMat  _C;
    Vec  _b;
    Vec  _u;
    Vec  _l;

    ///// model size
    isize _dim;
    isize _n_eq;
    isize _n_in;
    isize _n_total;

    Qpdata(isize dim, isize n_eq, isize n_in):
                    _H(dim, dim),
                    _g(dim),
                    _A(n_eq,dim),
                    _C(n_in,dim),
                    _b(n_eq),
                    _u(n_in),
                    _l(n_in)
    {
                    _dim = dim;
                    _n_eq = n_eq;
                    _n_in = n_in;
                    _n_total = _dim+_n_eq+_n_in;

                    _H.setZero();
                    _g.setZero();
                    _A.setZero();
                    _C.setZero();
                    _b.setZero();
                    _u.setZero();
                    _l.setZero();
    }
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPData_HPP_VCVSK3EOS */
