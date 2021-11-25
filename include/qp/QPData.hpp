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

	using VecMap = Eigen::Map<Vec const>;
	using VecMapMut = Eigen::Map<Vec>;

    using MatMap = Eigen::Map<ColMat const>;
	using MatMapMut = Eigen::Map<ColMat>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
    using VecISize = Eigen::Matrix<isize, DYN, 1>;

    using VecMapBool = Eigen::Map<Eigen::Matrix<bool, DYN, 1> const>;
    using VecBool = Eigen::Matrix<bool, DYN, 1>;

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

	Qpdata(ColMat H,Vec g, ColMat A, Vec b, ColMat C, Vec u, Vec l){   

                    _dim = H.rows();
                    _n_eq = A.rows();
                    _n_in = C.rows();
                    _n_total = _dim+_n_eq+_n_in;
                    _H(_dim, _dim);
                    _g(_dim);
                    _A(_n_eq,_dim);
                    _C(_n_in,_dim);
                    _b(_n_eq);
                    _u(_n_in);
                    _l(_n_in);
                    _H = H;
                    _g = g;
                    _A = A;
                    _C = C;
                    _b = b;
                    _u = u;
                    _l = l;
    }

};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPData_HPP_VCVSK3EOS */
