#ifndef INRIA_LDLT_QPResults_HPP_VCVSK3EOS
#define INRIA_LDLT_QPResults_HPP_VCVSK3EOS

#include <Eigen/Core>
//#include "ldlt/views.hpp"

namespace qp {

template <typename T>
struct QPResults {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// SOLUTION STORAGE

    Vec x;
    Vec y;
    Vec z;
    isize n_c; // final number of active inequalities

    ///// final proximal regularization parameters
    T mu_eq;
    T mu_eq_inv;
    T mu_in;
    T mu_in_inv;
    T rho;
    T nu;

    ///// iteration count
    isize n_tot;
    isize n_ext;
    isize n_mu_change;

    //// timings
    T timing;
    T objValue;

	QPResults( isize dim=0, isize n_eq=0, isize n_in=0)
			: //
                x(dim),
                y(n_eq),
                z(n_in)
                {
        
                x.setZero();
                y.setZero();
                z.setZero();

                rho = 1e-6;
	            mu_eq_inv = 1e-3;
	            mu_eq = 1e3 ;
	            mu_in_inv = 1e-1;
	            mu_in = 1e1;
                nu = 1.;

                n_tot = 0;
                n_ext = 0;
                n_mu_change = 0;
                n_c = 0;
                timing = 0.;
                objValue =0.;
                
                }
    
    void reset_results(){
        x.setZero();
        y.setZero();
        z.setZero();

        rho = 1e-6;
        mu_eq_inv = 1e-3;
        mu_eq = 1e3 ;
        mu_in_inv = 1e-1;
        mu_in = 1e1;
        nu = 1.;

        n_tot = 0;
        n_ext = 0;
        n_mu_change = 0;
        n_c = 0;
        timing = 0.;
        objValue =0.;

    }
};

} // namespace qp

#endif /* end of include guard INRIA_LDLT_QPResults_HPP_VCVSK3EOS */
