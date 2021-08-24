#include <doctest.h>
#include <ldlt/precond/ruiz.hpp>
#include <util.hpp>

using namespace ldlt;
using Scalar = long double;

DOCTEST_TEST_CASE("ruiz preconditioner") {
	i32 dim = 30;
	i32 n_eq = 6;
	Qp<Scalar> qp{random_with_dim_and_n_eq, dim, n_eq};
	Qp<Scalar> scaled_qp = qp;

	qp::preconditioner::RuizEquilibration<Scalar, colmajor, colmajor> precond{
			dim,
			n_eq,
	};

  {
    EigenNoAlloc _{};
    precond.scale_qp_in_place(scaled_qp.as_mut());
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
