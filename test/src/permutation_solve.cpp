#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <util.hpp>
#include <doctest.h>

using namespace ldlt;
DOCTEST_TEST_CASE("permute apply") {
	using T = f64;
	isize n = 13;
	auto in = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);
	auto rhs = ldlt_test::rand::vector_rand<T>(n);

	auto perm = std::vector<isize>(usize(n));
	auto perm_inv = std::vector<isize>(usize(n));
	ldlt::detail::compute_permutation<T>( //
			perm.data(),
			perm_inv.data(),
			{from_eigen, in.diagonal()});

	{
		auto l = in;
		auto d = Vec<T>(n);
		{
			LDLT_WORKSPACE_MEMORY(work, Uninit, Mat(n, n), LDLT_CACHELINE_BYTES, T);
			ldlt::detail::apply_permutation_sym_work<T>( //
					{from_eigen, l},
					perm.data(),
					work,
					-1);
		}
		auto ldl = LdltViewMut<T>{
				{from_eigen, l},
				{from_eigen, d},
		};
		ldlt::factorize(ldl, ldl.l.as_const());
		{
			LDLT_WORKSPACE_MEMORY(work_rhs, Uninit, Vec(n), LDLT_CACHELINE_BYTES, T);
			auto x = rhs;
			work_rhs.to_eigen() = x;
			ldlt::detail::apply_perm_rows<T>::fn(
					x.data(), 0, work_rhs.data, 0, n, 1, perm.data(), 0);

			ldlt::solve({from_eigen, x}, ldl.as_const(), {from_eigen, x});

			work_rhs.to_eigen() = x;
			ldlt::detail::apply_perm_rows<T>::fn(
					x.data(), 0, work_rhs.data, 0, n, 1, perm_inv.data(), 0);

			DOCTEST_CHECK(
					(in * x - rhs).norm() <= T(1e3) * std::numeric_limits<T>::epsilon());
		}
	}
}
