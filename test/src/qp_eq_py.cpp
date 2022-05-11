#include "cnpy.hpp"
#include <doctest.h>
#include <fstream>

#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <iostream>
#include <fmt/core.h>
#include <util.hpp>

using namespace proxsuite::qp;
using Scalar = double;

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
		
		isize n = isize(cnpy::npy_load_mat<Scalar>(path + "/H.npy").rows());
		isize n_eq = isize(cnpy::npy_load_mat<Scalar>(path + "/A.npy").rows());
		isize n_in = 0 ; 
		Scalar sparsity_factor = 0.15;
		//const auto C_= ldlt_test::rand::sparse_matrix_rand_not_compressed<Scalar>(0, n, sparsity_factor);
		Qp<Scalar> qp{random_with_dim_and_neq_and_n_in, n, n_eq, n_in, sparsity_factor};
	    //const auto u_ = ldlt_test::rand::vector_rand<Scalar>(0);
	    //const auto l_=  ldlt_test::rand::vector_rand<Scalar>(0);
		/*
		Qp<Scalar> qp{
				from_data,
				cnpy::npy_load_mat<Scalar>(path + "/H.npy"),
				cnpy::npy_load_vec<Scalar>(path + "/g.npy"),
				cnpy::npy_load_mat<Scalar>(path + "/A.npy"),
				cnpy::npy_load_vec<Scalar>(path + "/b.npy"),
				C_,
				u_,
				l_
		};
		*/
		qp.H = cnpy::npy_load_mat<Scalar>(path + "/H.npy");
		qp.g = cnpy::npy_load_vec<Scalar>(path + "/g.npy");
		qp.A = cnpy::npy_load_mat<Scalar>(path + "/A.npy");
		qp.b = cnpy::npy_load_vec<Scalar>(path + "/b.npy");
		//isize n_eq = isize(qp.A.rows());

		::fmt::print(
				"-- problem_path   : {}\n"
				"-- n              : {}\n"
				"-- n_eq           : {}\n",
				path,
				n,
				n_eq);

		qp::dense::QP<Scalar> Qp{n,n_eq,0}; // creating QP object
		Qp.settings.eps_abs = eps_abs;
		Qp.setup_dense_matrices(qp.H,qp.g,qp.A,qp.b,qp.C,qp.u,qp.l);
		Qp.solve();

		fmt::print(
				"-- iter           : {}\n"
				"-- primal residual: {}\n"
				"-- dual residual  : {}\n\n",
				Qp.results.info.iter,
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>(),
				(qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y)
						.lpNorm<Eigen::Infinity>());

		DOCTEST_CHECK(
				(qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
		DOCTEST_CHECK(
				(qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y)
						.lpNorm<Eigen::Infinity>() <= eps_abs);
	}
}
