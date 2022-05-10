#ifndef PROXSUITE_INCLUDE_QP_RESULTS_HPP
#define PROXSUITE_INCLUDE_QP_RESULTS_HPP

#include <Eigen/Core>
#include <veg/type_traits/core.hpp>
#include "qp/constants.hpp"

namespace qp {
using veg::isize;

template <typename T>
struct Info {
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
};

template <typename T>
struct Results {
public:
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	///// SOLUTION STORAGE

	Vec x;
	Vec y;
	Vec z;

	Info<T> info;

	////// SOLUTION STATUS

	Results(isize dim = 0, isize n_eq = 0, isize n_in = 0)
			: x(dim), y(n_eq), z(n_in) {

		x.setZero();
		y.setZero();
		z.setZero();

		info.rho = 1e-6;
		info.mu_eq_inv = 1e3;
		info.mu_eq = 1e-3;
		info.mu_in_inv = 1e1;
		info.mu_in = 1e-1;
		info.nu = 1.;
		info.iter = 0;
		info.iter_ext = 0;
		info.mu_updates = 0;
		info.rho_updates = 0;
		info.n_c = 0;
		info.run_time = 0;
		info.setup_time = 0;
		info.solve_time = 0;
		info.objValue = 0.;
		info.pri_res = 0.;
		info.dua_res = 0.;
		info.status = PROXQP_MAX_ITER_REACHED;
	}

	void reset_results() {
		x.setZero();
		y.setZero();
		z.setZero();

		info.rho = 1e-6;
		info.mu_eq_inv = 1e3;
		info.mu_eq = 1e-3;
		info.mu_in_inv = 1e1;
		info.mu_in = 1e-1;
		info.nu = 1.;
		info.iter = 0;
		info.iter_ext = 0;
		info.mu_updates = 0;
		info.rho_updates = 0;
		info.n_c = 0;
		info.run_time = 0;
		info.setup_time = 0;
		info.solve_time = 0;
		info.objValue = 0.;
		info.pri_res = 0.;
		info.dua_res = 0.;
		info.status = PROXQP_MAX_ITER_REACHED;
	}
};

} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_RESULTS_HPP */
