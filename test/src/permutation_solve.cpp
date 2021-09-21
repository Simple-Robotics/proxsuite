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

	auto perm = std::vector<i32>(usize(n));
	auto perm_inv = std::vector<i32>(usize(n));
	ldlt::detail::compute_permutation<T>( //
			perm.data(),
			perm_inv.data(),
			{from_eigen, in.diagonal()});

	{
		auto l = in;
		auto d = Vec<T>(n);
		{
			LDLT_WORKSPACE_MEMORY(work, Mat(n, n), T);
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
			LDLT_WORKSPACE_MEMORY(work_rhs, Vec(n), T);
			auto x = rhs;
			work_rhs.to_eigen() = x;
			ldlt::detail::apply_perm_rows<T>::fn(
					x.data(), 0, work_rhs.data, 0, 1, n, perm.data(), 0);

			ldlt::solve({from_eigen, x}, ldl.as_const(), {from_eigen, x});

			work_rhs.to_eigen() = x;
			ldlt::detail::apply_perm_rows<T>::fn(
					x.data(), 0, work_rhs.data, 0, 1, n, perm_inv.data(), 0);

			DOCTEST_CHECK(
					(in * x - rhs).norm() <= T(1e3) * std::numeric_limits<T>::epsilon());
		}
	}
}
