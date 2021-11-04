#include <ldlt/ldlt.hpp>
#include <util.hpp>
#include <doctest.h>

using namespace ldlt;

using T = f64;
TEST_CASE("delete") {
	for (isize n = 2; n < 32; ++n) {
		auto mat = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);

		for (isize i = 0; i < n; ++i) {
			auto mat_reduced = mat;
			isize rem = n - i - 1;

			mat_reduced.middleCols(i, rem) =
					mat_reduced.middleCols(i + 1, rem).eval();
			mat_reduced.conservativeResize(n, n - 1);
			mat_reduced.middleRows(i, rem) =
					mat_reduced.middleRows(i + 1, rem).eval();
			mat_reduced.conservativeResize(n - 1, n - 1);

			auto ldl = Ldlt<T>{decompose, mat};
			ldl.delete_at(i);

			CHECK((mat_reduced - ldl.reconstructed_matrix()).norm() < 1e3);
		}
	}
}
