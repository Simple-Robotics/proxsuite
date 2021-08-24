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

  std::ifstream file("/home/antoine/Bureau/thèse/projects/prox-qp/inria_ldlt/test/src/qp_problem/source_files.txt"); // to be changed 
  std::string path; 
  Scalar eps_abs = Scalar(1e-9);
  
  while (std::getline(file, path))
  {
    
    std::cout << path << std::endl<<std::endl;
    auto index = path.find("_n_eq");
    std::size_t index__ = path.find("dim_");

    auto n = std::stoi(  path.substr(index__+4,index - (index__+4) ) ) ; //c++11
    auto n_eq = std::stoi(  path.substr(index+6)  ); //c++11

    Qp<Scalar> qp{random_with_dim_and_n_eq, n, n_eq};

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_init(n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_init(n_eq);
    primal_init.setZero();
    primal_init = -qp.H.llt().solve(qp.g);
    dual_init.setZero();

    auto H = cnpy::npy_load_mat<double>(path + "/H.npy");
    auto g = cnpy::npy_load_vec<double>(path + "/g.npy");

    auto A = cnpy::npy_load_mat<double>(path + "/A.npy");
    auto b = cnpy::npy_load_vec<double>(path + "/b.npy");

    qp.H = H.cast<long double>();
    qp.A = A.cast<long double>();
    qp.b = b.cast<long double>();
    qp.g = g.cast<long double>();

	
	  auto iter = qp::detail::solve_qp( //
			detail::from_eigen_vector_mut(primal_init),
			detail::from_eigen_vector_mut(dual_init),
			qp.as_view(),
			200,
			eps_abs,
			0,
			qp::preconditioner::RuizEquilibration<Scalar, colmajor, colmajor>{
					n,
					n_eq,
			});

    std::cout << "-- iter : " << iter << std::endl << std::endl;
    std::cout << "-- primal residual : " << (qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() << std::endl << std::endl;
    std::cout << "-- dual residual : " << (qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() << std::endl << std::endl;
	  DOCTEST_CHECK(
			(qp.A * primal_init - qp.b).lpNorm<Eigen::Infinity>() <= eps_abs);
	  DOCTEST_CHECK(
			(qp.H * primal_init + qp.g + qp.A.transpose() * dual_init)
					.lpNorm<Eigen::Infinity>() <= eps_abs);

  }
}
