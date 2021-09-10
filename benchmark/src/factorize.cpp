#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <ldlt/factorize.hpp>
#include <util.hpp>

#define LDLT_BENCHMARK_MAIN BENCHMARK_MAIN    /* NOLINT */
#define LDLT_BENCHMARK BENCHMARK              /* NOLINT */
#define LDLT_BENCHMARK_TPL BENCHMARK_TEMPLATE /* NOLINT */

using namespace ldlt;

template <typename T, Layout L>
using Mat = Eigen::
		Matrix<T, -1, -1, (L == colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <typename T, Layout L>
void bench_eigen(benchmark::State& s) {
	isize dim = isize(s.range(0));

	Mat<T, L> a = ldlt_test::rand::positive_definite_rand<T>(dim, T(1e2));
	Eigen::LDLT<Mat<T, L>> l(a);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.matrixLDLT().data());

	for (auto _ : s) {
		l.compute(a);
		benchmark::ClobberMemory();
	}
}

template <typename Strategy, typename T, Layout L>
void bench_ours(benchmark::State& s) {

	isize dim = isize(s.range(0));
	Mat<T, L> a = ldlt_test::rand::positive_definite_rand<T>(dim, T(1e2));

	Mat<T, colmajor> l(dim, dim);
	l.setZero();
	Vec<T> d(dim);
	d.setZero();

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());

	for (auto _ : s) {
		auto a_view = MatrixView<T, L>{from_eigen, a};
		auto ldl_view = ldlt::LdltViewMut<T>{
				{from_eigen, l},
				{from_eigen, d},
		};

		factorize(ldl_view, a_view, Strategy{});
		benchmark::ClobberMemory();
	}
}

template <typename T>
void bench_ours_inplace(benchmark::State& s) {

	isize dim = isize(s.range(0));
	Mat<T, colmajor> a = ldlt_test::rand::positive_definite_rand<T>(dim, T(1e2));

	Mat<T, colmajor> l(dim, dim);
	l.setZero();
	Vec<T> d(dim);
	d.setZero();

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());

	for (auto _ : s) {
		auto a_view = MatrixView<T, colmajor>{from_eigen, a};
		auto ldl_view = ldlt::LdltViewMut<T>{
				{from_eigen, l},
				{from_eigen, d},
		};

		l = a;
		factorize(ldl_view, a_view);
		benchmark::ClobberMemory();
	}
}

void bench_dummy(benchmark::State& s) {
	for (auto _ : s) {
	}
}

LDLT_BENCHMARK_MAIN();

constexpr isize dim_small = 32;
constexpr isize dim_medium = 128;
constexpr isize dim_large = 1024;

namespace strat = factorization_strategy;

#define LDLT_BENCH(Dim, Type, L)                                               \
	LDLT_BENCHMARK(bench_dummy);                                                 \
	LDLT_BENCHMARK_TPL(bench_eigen, Type, L)->Arg(Dim);                          \
	LDLT_BENCHMARK_TPL(bench_ours, strat::Standard, Type, L)->Arg(Dim);

#define LDLT_BENCH_LAYOUT(Dim, Type)                                           \
	LDLT_BENCHMARK(bench_dummy);                                                 \
	LDLT_BENCH(Dim, Type, colmajor);                                             \
	LDLT_BENCH(Dim, Type, rowmajor);                                             \
	LDLT_BENCHMARK_TPL(bench_ours_inplace, Type)->Arg(Dim);

#define LDLT_BENCH_DIM(Type)                                                   \
	LDLT_BENCH_LAYOUT(dim_small, Type);                                          \
	LDLT_BENCH_LAYOUT(dim_medium, Type);                                         \
	LDLT_BENCH_LAYOUT(dim_large, Type)

LDLT_BENCH_DIM(f32);
LDLT_BENCH_DIM(f64);
