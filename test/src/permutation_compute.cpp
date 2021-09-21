#include <ldlt/factorize.hpp>
#include <util.hpp>
#include <algorithm>
#include <doctest.h>

using namespace ldlt;
DOCTEST_TEST_CASE_TEMPLATE("permute compute", T, f32, f64) {
	isize n = 13;
	auto m = ldlt_test::rand::matrix_rand<T>(n, n);
	auto perm = std::vector<i32>(usize(n));
	auto perm_inv = std::vector<i32>(usize(n));
	ldlt::detail::compute_permutation<T>( //
			perm.data(),
			perm_inv.data(),
			{from_eigen, m.diagonal()});
	auto tmp = Vec<T>(n);

	for (isize i = 0; i < n; ++i) {
		auto k = perm[usize(i)];
		tmp(i) = m.diagonal()(k);
	}

	DOCTEST_CHECK(std::is_sorted(tmp.data(), tmp.data() + n, [](T a, T b) {
		using std::fabs;
		return fabs(a) > fabs(b);
	}));
}
