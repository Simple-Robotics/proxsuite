#include <ldlt/ldlt.hpp>
#include <util.hpp>
#include <doctest.h>

using namespace ldlt;
DOCTEST_TEST_CASE("permute apply") {
	using T = f64;
	isize n = 13;
	using Mat = ::Mat<T, colmajor>;
	auto in = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);
	auto rhs = ldlt_test::rand::vector_rand<T>(n);

	Ldlt<T> ldl{decompose, in};
	auto sol = rhs;
	ldl.solve_in_place(sol);
	DOCTEST_CHECK((in * sol - rhs).norm() <= 1e3);

	auto l = ldl.l();
	auto lt = ldl.lt();
	auto d = ldl.d().asDiagonal();
	auto p = ldl.p();
	auto pt = ldl.pt();
	DOCTEST_CHECK((pt * Mat(l) * d * Mat(lt) * p - in).norm() <= 1e3);
}
