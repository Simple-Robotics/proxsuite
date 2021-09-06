#include <doctest.h>
#include <qp/precond/ruiz.hpp>
#include <util.hpp>
#include <iostream>

using namespace ldlt;
using Scalar = long double;

DOCTEST_TEST_CASE("ruiz preconditioner") {
	i32 dim = 5;
	i32 n_eq = 6;
	Scalar epsilon = Scalar(1.e-3);
	i32 max_iter = 20 ;
	i32 enum_ = 1; // 0 : upper triangular (by default), 1: lower triangular ; else full matrix

	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};
	if (enum_ == 0){
		qp.H = qp.H.triangularView<Eigen::Upper>();
	} else if (enum_ == 1){
		qp.H = qp.H.triangularView<Eigen::Lower>();
	}
	Qp<Scalar> scaled_qp = qp;
	std::cout << "qp.H : " << qp.H << std::endl << std::endl;
	qp::preconditioner::RuizEquilibration<Scalar, colmajor, colmajor> precond{
			dim,
			n_eq,
	};

  {
    EigenNoAlloc _{};
    precond.scale_qp_in_place(scaled_qp.as_mut(),epsilon,max_iter,enum_);
  }
	auto head = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
			precond.delta.head(dim).asDiagonal());
	auto tail = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
			precond.delta.tail(n_eq).asDiagonal());
	auto c = precond.c;

	auto const& H = qp.H;
	auto const& g = qp.g;
	auto const& A = qp.A;
	auto const& b = qp.b;

	auto H_new = (c * head * H * head).eval();
	auto g_new = (c * head * g).eval();
	auto A_new = (tail * A * head).eval();
	auto b_new = (tail * b).eval();

	DOCTEST_CHECK((H_new - scaled_qp.H).norm() <= Scalar(1e-10));
	DOCTEST_CHECK((g_new - scaled_qp.g).norm() <= Scalar(1e-10));
	DOCTEST_CHECK((A_new - scaled_qp.A).norm() <= Scalar(1e-10));
	DOCTEST_CHECK((b_new - scaled_qp.b).norm() <= Scalar(1e-10));
}
