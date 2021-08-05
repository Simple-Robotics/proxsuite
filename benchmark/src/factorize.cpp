#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <ldlt/ldlt.hpp>

#define LDLT_BENCHMARK_MAIN BENCHMARK_MAIN    /* NOLINT */
#define LDLT_BENCHMARK BENCHMARK              /* NOLINT */
#define LDLT_BENCHMARK_TPL BENCHMARK_TEMPLATE /* NOLINT */

using namespace ldlt;

template <typename T, Layout L>
using Mat = Eigen::
		Matrix<T, -1, -1, (L == colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <typename T, Layout InL, Layout OutL>
void bench_eigen(benchmark::State& s) {
	i32 dim = i32(s.range(0));

	Mat<T, InL> a(dim, dim);
	{
		a.setRandom();
		a = a.transpose() * a;
	}

	Eigen::LDLT<Mat<T, OutL>> l(dim);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(
			const_cast /* NOLINT(cppcoreguidelines-pro-type-const-cast) */<T*>(
					l.matrixLDLT().data()));

	for (auto _ : s) {
		l.compute(a);
		benchmark::ClobberMemory();
	}
}

template <typename LdltFn, typename T, Layout InL, Layout OutL>
void bench_ours(benchmark::State& s) {

	i32 dim = i32(s.range(0));
	Mat<T, InL> a(dim, dim);
	{
		a.setRandom();
		a = a.transpose() * a;
	}

	Mat<T, OutL> l(dim, dim);
	l.setZero();
	Vec<T> d(dim);
	d.setZero();

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());

	for (auto _ : s) {
		auto a_view = ldlt::MatrixView<T, InL>{a.data(), dim};
		auto l_view = ldlt::MatrixViewMut<T, OutL>{l.data(), dim};
		auto d_view = ldlt::DiagonalMatrixViewMut<T>{d.data(), dim};

		LdltFn{}(l_view, d_view, a_view);
		benchmark::ClobberMemory();
	}
}

void bench_dummy(benchmark::State& s) {
	for (auto _ : s) {
	}
}

LDLT_BENCHMARK_MAIN();

constexpr i32 dim_small = 32;
constexpr i32 dim_medium = 128;
constexpr i32 dim_large = 1024;

#define LDLT_BENCH(Dim, Type, InL, OutL)                                       \
	LDLT_BENCHMARK(bench_dummy);                                                 \
	LDLT_BENCHMARK_TPL(bench_eigen, Type, InL, OutL)->Arg(Dim);                  \
	LDLT_BENCHMARK_TPL(bench_ours, ldlt::nb::factorize, Type, InL, OutL)         \
			->Arg(Dim);                                                              \
	LDLT_BENCHMARK_TPL(                                                          \
			bench_ours, ldlt::nb::factorize_defer_to_colmajor, Type, InL, OutL)      \
			->Arg(Dim)

#define LDLT_BENCH_LAYOUT(Dim, Type)                                           \
	LDLT_BENCH(Dim, Type, colmajor, colmajor);                                   \
	LDLT_BENCH(Dim, Type, rowmajor, colmajor);                                   \
	LDLT_BENCH(Dim, Type, colmajor, rowmajor);                                   \
	LDLT_BENCH(Dim, Type, rowmajor, rowmajor)

#define LDLT_BENCH_DIM(Type)                                                   \
	LDLT_BENCH_LAYOUT(dim_small, Type);                                          \
	LDLT_BENCH_LAYOUT(dim_medium, Type);                                         \
	LDLT_BENCH_LAYOUT(dim_large, Type)

LDLT_BENCH_DIM(f32);
LDLT_BENCH_DIM(f64);
