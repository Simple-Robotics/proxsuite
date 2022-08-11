//
// Copyright (c) 2022, INRIA
//
#include "cnpy.hpp"
#include <doctest.h>
#include <fstream>

#include <doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <qp/dense/dense.hpp>
#include <iostream>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

using namespace proxsuite::qp;
using Scalar = double;

DOCTEST_TEST_CASE("qp: test qp loading and solving")
{

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
    isize n_in = 0;
    Scalar sparsity_factor = 0.15;
    Scalar strong_convexity_factor = 0.01;
    proxqp::dense::Model<T> qp = proxqp::utils::dense_strongly_convex_qp(
      dim, n_eq, n_in, sparsity_factor, strong_convexity_factor);
    qp.H = cnpy::npy_load_mat<Scalar>(path + "/H.npy");
    qp.g = cnpy::npy_load_vec<Scalar>(path + "/g.npy");
    qp.A = cnpy::npy_load_mat<Scalar>(path + "/A.npy");
    qp.b = cnpy::npy_load_vec<Scalar>(path + "/b.npy");
    std::cout << "-- problem_path   : " << path << std::endl;
    std::cout << "-- n   : " << n << std::endl;
    std::cout << "-- n_eq   : " << n_eq << std::endl;

    qp::dense::QP<Scalar> Qp{ n, n_eq, 0 }; // creating QP object
    Qp.settings.eps_abs = eps_abs;
    Qp.init(qp.H, qp.g, qp.A, qp.b, qp.C, qp.u, qp.l);
    Qp.solve();
    std::cout << "-- iter           : " << Qp.results.info.iter << std::endl;
    std::cout << "-- primal residual: "
              << (qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>()
              << std::endl;
    std::cout << "-- dual residual  : "
              << (qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y)
                   .lpNorm<Eigen::Infinity>()
              << std::endl;

    DOCTEST_CHECK((qp.A * Qp.results.x - qp.b).lpNorm<Eigen::Infinity>() <=
                  eps_abs);
    DOCTEST_CHECK((qp.H * Qp.results.x + qp.g + qp.A.transpose() * Qp.results.y)
                    .lpNorm<Eigen::Infinity>() <= eps_abs);
  }
}
