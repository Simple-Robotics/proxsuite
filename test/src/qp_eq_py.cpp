#include "cnpy.hpp"
#include <doctest.h>
#include <fstream>

#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <ldlt/qp/eq_solver.hpp>
#include <ldlt/precond/ruiz.hpp>
#include <iostream>
#include <fmt/core.h>
#include <util.hpp>

using namespace ldlt;
using Scalar = long double;

DOCTEST_TEST_CASE("qp: test qp loading and solving") {

	std::ifstream file(INRIA_LDLT_QP_PYTHON_PATH
	                   "source_files.txt"); // to be changed
	Scalar eps_abs = Scalar(1e-9);

	std::string path_len;
	std::string path;
	path_len.resize(32);
	while (true) {
		file.read(&path_len[0], 32);
		file.get(); // ':'
		i32 ipath_len = i32(std::stol(path_len));
		path.resize(usize(ipath_len));
		file.read(&path[0], ipath_len);
		file.get(); // '\n'

		Qp<Scalar> qp{
				from_data,
				cnpy::npy_load_mat<double>(path + "/H.npy").cast<long double>(),
				cnpy::npy_load_vec<double>(path + "/g.npy").cast<long double>(),
				cnpy::npy_load_mat<double>(path + "/A.npy").cast<long double>(),
				cnpy::npy_load_vec<double>(path + "/b.npy").cast<long double>(),
		};
		i32 n = i32(qp.H.rows());
		i32 n_eq = i32(qp.A.rows());

		using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
		Vec primal_init = Vec::Zero(n);
		Vec dual_init = Vec::Zero(n_eq);

		fmt::print(
				"-- problem_path   : {}\n"
				"-- n              : {}\n"
				"-- n_eq           : {}\n",
				path,
				n,
				n_eq);

		auto iter = [&] {
			auto ruiz =
					qp::preconditioner::RuizEquilibration<Scalar, colmajor, colmajor>{
							n,
							n_eq,
					};
			EigenNoAlloc _{};
			return qp::detail::solve_qp( //
					detail::from_eigen_vector_mut(primal_init),
					detail::from_eigen_vector_mut(dual_init),
					qp.as_view(),
					200,
					eps_abs,
					0,
					LDLT_FWD(ruiz));
		}();

		fmt::print(
				"-- iter           : {}\n"
				"-- primal residual: {}\n"
				"-- dual residual  : {}\n\n",
				iter,
				(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>(),
				(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
						.lpNorm<Eigen::Infinity>());

		DOCTEST_CHECK(
				(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
		DOCTEST_CHECK(
				(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
						.lpNorm<Eigen::Infinity>() <= eps_abs);
	}
}
