#ifndef INRIA_LDLT_QPResults_HPP_VCVSK3EOS
#define INRIA_LDLT_QPResults_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"

namespace qp {

template <typename T>
struct Qpresults {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// SOLUTION STORAGE

    Vec _x;
    Vec _y;
    Vec _z;
    isize _n_c; // final number of active inequalities

    ///// final proximal regularization parameters
    T _mu_eq;
    T _mu_eq_inv;
    T _mu_in;
    T _mu_in_inv;
    T _rho;

    ///// iteration count
    isize _n_tot;
    isize _n_ext;
    isize _n_mu_change;

    //// timings
    T _timing;

	Qpresults( isize dim=0, isize n_eq=0, isize n_in=0)
			: //
                _x(dim),
                _y(n_eq),
                _z(n_in)
                {
        
                _x.setZero();
                _y.setZero();
                _z.setZero();

                _rho = 1e-6;
	            _mu_eq_inv = 1e-3;
	            _mu_eq = 1e3 ;
	            _mu_in_inv = 1e-1;
	            _mu_in = 1e1;

                _n_tot = 0;
                _n_ext = 0;
                _n_mu_change = 0;
                _n_c = 0;
                _timing = 0.;
                }
    
    void clearResults(){
        _x.setZero();
        _y.setZero();
        _z.setZero();

        _rho = 1e-6;
        _mu_eq_inv = 1e-3;
        _mu_eq = 1e3 ;
        _mu_in_inv = 1e-1;
        _mu_in = 1e1;

        _n_tot = 0;
        _n_ext = 0;
        _n_mu_change = 0;
        _n_c = 0;
        _timing = 0.;
    }
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPResults_HPP_VCVSK3EOS */
