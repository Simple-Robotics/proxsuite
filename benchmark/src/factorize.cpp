#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <ldlt/ldlt.hpp>

#define LDLT_BENCHMARK_MAIN BENCHMARK_MAIN    /* NOLINT */
#define LDLT_BENCHMARK BENCHMARK              /* NOLINT */
#define LDLT_BENCHMARK_TPL BENCHMARK_TEMPLATE /* NOLINT */

using namespace ldlt;

template <typename T>
using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <typename T>
void bench_eigen(benchmark::State& s) {
	i32 dim = i32(s.range(0));

	Mat<T> a(dim, dim);
	{
		a.setRandom();
		a = a.transpose() * a;
	}

	Eigen::LDLT<Mat<T>> l(dim);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(
			const_cast /* NOLINT(cppcoreguidelines-pro-type-const-cast) */<T*>(
					l.matrixLDLT().data()));

	for (auto _ : s) {
		l.compute(a);
		benchmark::ClobberMemory();
	}
}

template <typename T>
void bench_ours(benchmark::State& s) {

	i32 dim = i32(s.range(0));
	Mat<T> a(dim, dim);
	{
		a.setRandom();
		a = a.transpose() * a;
	}

	Mat<T> l(dim, dim);
	l.setZero();
	Vec<T> d(dim);
	d.setZero();

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());

	for (auto _ : s) {
		auto a_view = ldlt::MatrixView<T, colmajor>{a.data(), dim};
		auto l_view =
				ldlt::LowerTriangularMatrixViewMut<T, colmajor>{l.data(), dim};
		auto d_view = ldlt::DiagonalMatrixViewMut<T>{d.data(), dim};

		ldlt::factorize_ldlt_unblocked(l_view, d_view, a_view);
		benchmark::ClobberMemory();
	}
}

LDLT_BENCHMARK_MAIN();

constexpr i32 dim_small = 32;
constexpr i32 dim_medium = 128;
constexpr i32 dim_large = 1024;

#define LDLT_ALL_BENCH(dim)                                                    \
	LDLT_BENCHMARK_TPL(bench_eigen, f32)->Arg(dim);                              \
	LDLT_BENCHMARK_TPL(bench_ours, f32)->Arg(dim);                               \
                                                                               \
	LDLT_BENCHMARK_TPL(bench_eigen, f64)->Arg(dim);                              \
	LDLT_BENCHMARK_TPL(bench_ours, f64)->Arg(dim)

LDLT_ALL_BENCH(dim_small);
LDLT_ALL_BENCH(dim_medium);
LDLT_ALL_BENCH(dim_large);
