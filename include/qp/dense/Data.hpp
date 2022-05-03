#ifndef PROXSUITE_INCLUDE_QP_DENSE_DATA_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_DATA_HPP

#include <Eigen/Core>

namespace qp {

namespace dense {
template <typename T>
struct Data {
public:
	static constexpr auto DYN = Eigen::Dynamic;
    enum { layout = Eigen::RowMajor };
	using Mat = Eigen::Matrix<T, DYN, DYN, layout>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// QP STORAGE
    Mat H;
    Vec  g;
    Mat  A;
    Mat  C;
    Vec  b;
    Vec  u;
    Vec  l;

    ///// model size
    isize dim;
    isize n_eq;
    isize n_in;
    isize n_total;

    Data(isize _dim, isize _n_eq, isize _n_in):
                    H(_dim, _dim),
                    g(_dim),
                    A(_n_eq,_dim),
                    C(_n_in,_dim),
                    b(_n_eq),
                    u(_n_in),
                    l(_n_in)
    {
                    dim = _dim;
                    n_eq = _n_eq;
                    n_in = _n_in;
                    n_total = _dim+_n_eq+_n_in;

                    H.setZero();
                    g.setZero();
                    A.setZero();
                    C.setZero();
                    b.setZero();
                    u.setZero();
                    l.setZero();
    }
};

} // namespace dense
} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_DATA_HPP */
