#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <osqp.h>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <ldlt/qp/eq_solver.hpp>
#include <ldlt/precond/ruiz.hpp>

using namespace ldlt;

template <typename Scalar>
using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, c_int>;

template <typename Scalar>
using Mat =
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename Scalar>
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using Scalar = double;

namespace ldlt_test {
namespace rand {

using std::uint64_t;

using uint128_t = __uint128_t;
uint128_t g_lehmer64_state =
		uint128_t(0xda942042e4dd58b5) * uint128_t(0xda942042e4dd58b5);

auto lehmer64() -> uint64_t { // [0, 2^64)
	g_lehmer64_state *= 0xda942042e4dd58b5;
	return g_lehmer64_state >> 64U;
}

void set_seed(uint64_t seed) {
	g_lehmer64_state = seed + 1;
	lehmer64();
	lehmer64();
}

auto uniform_rand() -> double { // [0, 2^53]
	uint64_t a = lehmer64() / (1U << 11U);
	return double(a) / double(uint64_t(1) << 53U);
}
auto normal_rand() -> double {
	static const double pi2 = std::atan(static_cast<double>(1)) * 8;

	double u1 = uniform_rand();
	double u2 = uniform_rand();

	double ln = std::log(u1);
	double sqrt = std::sqrt(-2 * ln);

	return sqrt * std::cos(pi2 * u2);
}

template <typename Scalar>
auto sparse_positive_definite_rand(i32 n, Scalar cond, double p)
		-> SparseMat<Scalar> {
	auto H = SparseMat<Scalar>(n, n);

	for (i32 i = 0; i < n; ++i) {
		auto urandom = rand::uniform_rand();
		if (urandom < p) {
			auto random = Scalar(rand::normal_rand());
			H.insert(i, i) = random;
		}
	}

	for (i32 i = 0; i < n; ++i) {
		for (i32 j = i + 1; j < n; ++j) {
			auto urandom = rand::uniform_rand();
			if (urandom < p / 2) {
				auto random = Scalar(rand::normal_rand());
				H.insert(i, j) = random;
			}
		}
	}

	using Mat =
			Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	Mat H_dense = H.toDense();
	Vec eigh = H_dense.template selfadjointView<Eigen::Upper>().eigenvalues();

	Scalar min = eigh.minCoeff();
	Scalar max = eigh.maxCoeff();

	// new_min = min + rho
	// new_max = max + rho
	//
	// (max + rho)/(min + rho) = cond
	// 1 + (max - min) / (min + rho) = cond
	// (max - min) / (min + rho) = cond - 1
	// min + rho = (max - min) / (cond - 1)
	// rho = (max - min)/(cond - 1) - min
	Scalar rho = (max - min) / (cond - 1) - min;

	for (i32 i = 0; i < n; ++i) {
		H.coeffRef(i, i) += rho;
	}
	H.makeCompressed();
	return H;
}

template <typename Scalar>
auto sparse_matrix_rand(i32 nrows, i32 ncols, double p) -> SparseMat<Scalar> {
	auto A = SparseMat<Scalar>(nrows, ncols);

	for (i32 i = 0; i < nrows; ++i) {
		for (i32 j = 0; j < ncols; ++j) {
			if (rand::uniform_rand() < p) {
				A.insert(i, j) = Scalar(rand::normal_rand());
			}
		}
	}
	A.makeCompressed();
	return A;
}

template <typename Scalar>
auto vector_rand(i32 nrows) -> Vec<Scalar> {
	auto v = Vec<Scalar>(nrows);

	for (i32 i = 0; i < nrows; ++i) {
		v(i) = Scalar(rand::normal_rand());
	}

	return v;
}
} // namespace rand

template <typename Scalar>
auto eigen_to_osqp_mat(SparseMat<Scalar>& mat) -> csc {
	return {
			mat.nonZeros(),
			mat.rows(),
			mat.cols(),
			mat.outerIndexPtr(),
			mat.innerIndexPtr(),
			mat.valuePtr(),
			-1,
	};
}
} // namespace ldlt_test

auto main() -> int {
	i32 dim = 1000;
	i32 n_eq = 100;

	double p = 1;
	auto cond = Scalar(1e2);

	auto H_eigen = ldlt_test::rand::sparse_positive_definite_rand(dim, cond, p);
	auto A_eigen = ldlt_test::rand::sparse_matrix_rand<Scalar>(n_eq, dim, p);
	auto g_eigen = ldlt_test::rand::vector_rand<Scalar>(dim);
	auto b_eigen = ldlt_test::rand::vector_rand<Scalar>(n_eq);

	auto H = ldlt_test::eigen_to_osqp_mat(H_eigen);
	auto A = ldlt_test::eigen_to_osqp_mat(A_eigen);
	auto osqp = OSQPData{
			dim,
			n_eq,
			&H,
			&A,
			g_eigen.data(),
			b_eigen.data(),
			b_eigen.data(),
	};

	OSQPSettings osqp_settings{};
	osqp_set_default_settings(&osqp_settings);
	osqp_settings.eps_rel = 1e-9;
	osqp_settings.eps_abs = 1e-9;

	OSQPWorkspace* osqp_work{};

	{
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp total"); }
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp solve"); }
		{ LDLT_DECL_SCOPE_TIMER("osqp bench", "ours"); }
		(void)0;
	}

	{
		LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp total");
		osqp_setup(&osqp_work, &osqp, &osqp_settings);
		{
			LDLT_DECL_SCOPE_TIMER("osqp bench", "osqp solve");
			osqp_solve(osqp_work);
		}
	}

	auto x = Eigen::Map<Vec<Scalar>>(osqp_work->solution->x, dim);
	auto y = Eigen::Map<Vec<Scalar>>(osqp_work->solution->y, n_eq);

	fmt::print(
			"{} {}\n",
			(H_eigen.selfadjointView<Eigen::Upper>() * x + g_eigen +
	     A_eigen.transpose() * y)
					.lpNorm<Eigen::Infinity>(),
			(A_eigen * x - b_eigen).lpNorm<Eigen::Infinity>());

	auto n_iters = [&] {
		x.setZero();
		y.setZero();

		auto H = Mat<Scalar>(H_eigen.toDense().selfadjointView<Eigen::Upper>());
		auto A = Mat<Scalar>(A_eigen.toDense());
		auto ruiz = qp::preconditioner::IdentityPrecond{};

		LDLT_DECL_SCOPE_TIMER("osqp bench", "ours");

		return qp::detail::solve_qp(
							 VectorViewMut<Scalar>{x.data(), i32(x.rows())},
							 VectorViewMut<Scalar>{y.data(), i32(y.rows())},
							 qp::QpView<Scalar, colmajor, colmajor>{
									 detail::from_eigen_matrix(H),
									 detail::from_eigen_vector(g_eigen),
									 detail::from_eigen_matrix(A),
									 detail::from_eigen_vector(b_eigen),
									 {},
									 {},
							 },
							 1000,
							 1e-9,
							 0,
							 LDLT_FWD(ruiz))
		    .n_iters;
	}();
	fmt::print("{}\n", n_iters);

	fmt::print(
			"{} {}\n",
			(H_eigen.selfadjointView<Eigen::Upper>() * x + g_eigen +
	     A_eigen.transpose() * y)
					.lpNorm<Eigen::Infinity>(),
			(A_eigen * x - b_eigen).lpNorm<Eigen::Infinity>());

	for (auto& outer : LDLT_GET_MAP()["osqp bench"]) {
		fmt::print("{}: {}\n", outer.first, outer.second.ref);
	}
	for (auto& outer : LDLT_GET_MAP(Scalar)["eq solver"]) {
		fmt::print("{}: {}\n", outer.first, outer.second.ref);
	}

	osqp_cleanup(osqp_work);
}
