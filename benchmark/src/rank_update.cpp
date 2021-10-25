#include <benchmark/benchmark.h>
#include <Eigen/Cholesky>
#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

#define LDLT_BENCHMARK_MAIN BENCHMARK_MAIN    /* NOLINT */
#define LDLT_BENCHMARK BENCHMARK              /* NOLINT */
#define LDLT_BENCHMARK_TPL BENCHMARK_TEMPLATE /* NOLINT */

using namespace ldlt;

using T = f64;

void bench_ours___(benchmark::State& s) {
	isize n = isize(s.range(0));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	Mat<T, colmajor> l(n, n);
	detail::set_zero(l.data(), usize(l.size()));
	Vec<T> d(n);
	detail::set_zero(d.data(), usize(d.size()));

	auto ldl = LdltViewMut<T>{{from_eigen, l}, {from_eigen, d}};
	factorize(ldl, MatrixView<T, colmajor>{from_eigen, a});

	Vec<T> z(n);
	z.setConstant(-0.0);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());
	benchmark::DoNotOptimize(z.data());

	for (auto _ : s) {
		rank1_update(ldl, ldl.as_const(), {from_eigen, z}, 1.0);
		benchmark::ClobberMemory();
	}
}

void bench_ours__r(benchmark::State& s) {
	isize n = isize(s.range(0));
	isize r = isize(s.range(1));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	Mat<T, colmajor> l(n, n);
	detail::set_zero(l.data(), usize(l.size()));
	Vec<T> d(n);
	detail::set_zero(d.data(), usize(d.size()));

	auto ldl = LdltViewMut<T>{{from_eigen, l}, {from_eigen, d}};
	factorize(ldl, MatrixView<T, colmajor>{from_eigen, a});

	Mat<T, colmajor> z(n, r);
	z.setConstant(-0.0);
	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());
	benchmark::DoNotOptimize(z.data());

	LDLT_MULTI_WORKSPACE_MEMORY(
			(z_tmp, Uninit, Mat(n, r), LDLT_CACHELINE_BYTES, T),
			(a_tmp, Uninit, Vec(r), LDLT_CACHELINE_BYTES, T));

	for (auto _ : s) {
		z_tmp.to_eigen() = z;
		a_tmp.to_eigen() = a;
		detail::rank_r_update(
				ldl, ldl.as_const(), {from_eigen, z}, {from_eigen, alpha});
		benchmark::ClobberMemory();
	}
}
void bench_ours_r4(benchmark::State& s) {
	isize n = isize(s.range(0));
	isize r = isize(s.range(1));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	Mat<T, colmajor> l(n, n);
	detail::set_zero(l.data(), usize(l.size()));
	Vec<T> d(n);
	detail::set_zero(d.data(), usize(d.size()));

	auto ldl = LdltViewMut<T>{{from_eigen, l}, {from_eigen, d}};
	factorize(ldl, MatrixView<T, colmajor>{from_eigen, a});

	Mat<T, colmajor> z(n, r);
	z.setConstant(-0.0);
	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(l.data());
	benchmark::DoNotOptimize(d.data());
	benchmark::DoNotOptimize(z.data());

	LDLT_MULTI_WORKSPACE_MEMORY(
			(z_tmp, Uninit, Mat(n, r), LDLT_CACHELINE_BYTES, T),
			(a_tmp, Uninit, Vec(r), LDLT_CACHELINE_BYTES, T));

	for (auto _ : s) {
		z_tmp.to_eigen() = z;
		a_tmp.to_eigen() = a;

		isize i = 0;
		while (true) {
			if (i == r) {
				break;
			}

			isize r_block = ldlt::detail::min2(r - i, isize(4));
			detail::rank_r_update(
					ldl,
					ldl.as_const(),
					z_tmp.block(0, i, z_tmp.rows, r_block),
					a_tmp.segment(i, r_block));
			i += r_block;
		}
		benchmark::ClobberMemory();
	}
}

void bench_eigen__(benchmark::State& s) {
	isize n = isize(s.range(0));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	auto ldl = a.ldlt();

	Vec<T> z(n);
	z.setConstant(-0.0);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(const_cast<T*>(ldl.matrixLDLT().data()));

	for (auto _ : s) {
		ldl.rankUpdate(z, 1.0);
		benchmark::ClobberMemory();
	}
}
void bench_eigen_r(benchmark::State& s) {
	isize n = isize(s.range(0));
	isize r = isize(s.range(1));

	ldlt_test::rand::set_seed(0);
	auto a = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));

	auto ldl = a.ldlt();

	Mat<T, colmajor> z(n, r);
	z.setConstant(-0.0);
	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

	benchmark::DoNotOptimize(a.data());
	benchmark::DoNotOptimize(const_cast<T*>(ldl.matrixLDLT().data()));

	for (auto _ : s) {
		for (isize i = 0; i < r; ++i) {
			ldl.rankUpdate(z.col(i), alpha(i));
		}
		benchmark::ClobberMemory();
	}
}

void args_1(benchmark::internal::Benchmark* b) {
	isize ns[] = {64, 128, 255, 256, 1024};
	for (auto n : ns) {
		b->Arg(n);
	}
}
void args_2(benchmark::internal::Benchmark* b) {
	isize ns[] = {64, 128, 255, 256, 1024};
	isize rs[] = {1, 2, 3, 4, 8, 16, 32};
	for (auto n : ns) {
		for (auto r : rs) {
			b->Args({n, r});
		}
	}
}

LDLT_BENCHMARK(bench_ours_r4)->Apply(args_2);
LDLT_BENCHMARK(bench_ours__r)->Apply(args_2);
LDLT_BENCHMARK(bench_eigen_r)->Apply(args_2);
LDLT_BENCHMARK(bench_ours___)->Apply(args_1);
LDLT_BENCHMARK(bench_eigen__)->Apply(args_1);

LDLT_BENCHMARK_MAIN();
