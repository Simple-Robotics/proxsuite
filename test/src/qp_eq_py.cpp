#include "cnpy.hpp"
#include <doctest.h>
#include <fstream>

#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/eq_solver.hpp>
#include <qp/precond/ruiz.hpp>
#include <iostream>
#include <fmt/core.h>
#include <util.hpp>

using namespace qp;
using Scalar = long double;

DOCTEST_TEST_CASE("qp: test qp loading and solving") {

	std::ifstream file(INRIA_LDLT_QP_PYTHON_PATH
	                   "source_files.txt"); // to be changed
	Scalar eps_abs = Scalar(1e-9);

	std::string path_len;
	std::string path;
	path_len.resize(32);
	file.read(&path_len[0], 32);
	file.get(); // '\n'
	i64 n_files = i64(std::stoll(path_len));

	for (i64 i = 0; i < n_files; ++i) {
		file.read(&path_len[0], 32);
		file.get(); // ':'
		isize ipath_len = isize(std::stoll(path_len));
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
		isize n = isize(qp.H.rows());
		isize n_eq = isize(qp.A.rows());

		using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
		Vec primal_init = Vec::Zero(n);
		Vec dual_init = Vec::Zero(n_eq);

		auto iter = [&] {
			auto ruiz = qp::preconditioner::RuizEquilibration<Scalar>{
					n,
					n_eq,
			};
			EigenNoAlloc _{};
			return qp::detail::solve_qp( //
                                   {qp::from_eigen, primal_init},
                                   {qp::from_eigen, dual_init},
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
				iter.n_iters,
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
