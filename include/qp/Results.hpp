#ifndef PROXSUITE_INCLUDE_QP_RESULTS_HPP
#define PROXSUITE_INCLUDE_QP_RESULTS_HPP

#include <Eigen/Core>
#include <veg/type_traits/core.hpp>
#include "qp/constants.hpp"

namespace qp {
using veg::isize;

template <typename T>
struct Results {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	///// SOLUTION STORAGE

	Vec x;
	Vec y;
	Vec z;

	///// final proximal regularization parameters
	T mu_eq;
	T mu_eq_inv;
	T mu_in;
	T mu_in_inv;
	T rho;
	T nu;

	///// iteration count
	isize iter;
	isize iter_ext;
	isize mu_updates;
    isize rho_updates;
    isize n_c; // final number of active inequalities
    isize status;

	//// timings
    T setup_time;
    T solve_time;
	T run_time;
	T objValue;
    T pri_res;
    T dua_res;

    ////// SOLUTION STATUS

	Results(isize dim = 0, isize n_eq = 0, isize n_in = 0)
			: //
                x(dim),
                y(n_eq),
                z(n_in)
                {
        
                x.setZero();
                y.setZero();
                z.setZero();

                rho = 1e-6;
	            mu_eq_inv = 1e3;
	            mu_eq = 1e-3 ;
	            mu_in_inv = 1e1;
	            mu_in = 1e-1;
                nu = 1.;

                iter = 0;
                iter_ext = 0;
                mu_updates = 0;
                rho_updates = 0;
                n_c = 0;
                run_time = 0.;
                setup_time = 0.;
                solve_time =0.;
                objValue =0.;
                pri_res = 0.;
                dua_res = 0.;

				status = PROXQP_MAX_ITER_REACHED;
                
                }
    
    void reset_results(){
        x.setZero();
        y.setZero();
        z.setZero();

        rho = 1e-6;
        mu_eq_inv = 1e3;
        mu_eq = 1e-3 ;
        mu_in_inv = 1e1;
        mu_in = 1e-1;
        nu = 1.;

        iter = 0;
        iter_ext = 0;
        mu_updates = 0;
        rho_updates = 0;
        n_c = 0;
        run_time = 0.;
        setup_time = 0.;
        solve_time =0.;
        objValue =0.;

		status = PROXQP_MAX_ITER_REACHED;

    }
};

} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_RESULTS_HPP */
