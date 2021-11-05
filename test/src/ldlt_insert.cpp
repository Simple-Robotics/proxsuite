#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>
#include <ldlt/ldlt.hpp>
#include "ldlt/views.hpp"
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using namespace ldlt;
using T = f64;

DOCTEST_TEST_CASE("insert") {

	using namespace ldlt::tags;

	T const eps = std::numeric_limits<T>::epsilon() * T(1e3);

	for (isize n = 1; n < 32; ++n) {
		auto mat = ldlt_test::rand::positive_definite_rand<T>(n, 1e2);
		auto col = ldlt_test::rand::vector_rand<T>(n + 1);
		for (isize i = 0; i < n + 1; ++i) {
			auto mat_target = [&] {
				auto mat_target = Mat<T, colmajor>(n + 1, n + 1);
				isize rem = n - i;

				mat_target.topLeftCorner(i, i) = mat.topLeftCorner(i, i);
				mat_target.bottomLeftCorner(rem, i) = mat.bottomLeftCorner(rem, i);
				mat_target.topRightCorner(i, rem) = mat.topRightCorner(i, rem);
				mat_target.bottomRightCorner(rem, rem) =
						mat.bottomRightCorner(rem, rem);

				mat_target.col(i) = col;
				mat_target.row(i) = col.transpose();

				return mat_target;
			}();

			auto ldl = ldlt::Ldlt<T>{decompose, mat};
			ldl.insert_at(i, col);

			DOCTEST_CHECK((mat_target - ldl.reconstructed_matrix()).norm() <= eps);
		}
	}
}

DOCTEST_TEST_CASE("append Maros example HS118") {

	using namespace ldlt::tags;

	isize n = 15;
	T eps = T(1e-10);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_m_init, Init, Mat(n, n), LDLT_CACHELINE_BYTES, T),
			(_m_to_get_, Init, Mat(n + 1, n + 1), LDLT_CACHELINE_BYTES, T),
			(row_, Init, Vec(n + 1), LDLT_CACHELINE_BYTES, T));

	auto m_init = _m_init.to_eigen();
	auto m_to_get = _m_to_get_.to_eigen();
	auto row = row_.to_eigen();

	m_init.diagonal() << 0.000201, 0.000201, 0.000301, 0.000201, 0.000201,
			0.000301, 0.000201, 0.000201, 0.000301, 0.000201, 0.000201, 0.000301,
			0.000201, 0.000201, 0.000301;

	ldlt::Ldlt<T> ldl{decompose, m_init};

	DOCTEST_CHECK((m_init - ldl.reconstructed_matrix()).norm() <= eps);

	row << 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1;

	ldl.insert_at(n, row);

	m_to_get.diagonal() << 0.000201, 0.000201, 0.000301, 0.000201, 0.000201,
			0.000301, 0.000201, 0.000201, 0.000301, 0.000201, 0.000201, 0.000301,
			0.000201, 0.000201, 0.000301, -0.1;

	m_to_get.block(n, 0, 1, n + 1) = row.transpose(); // insert at the end
	m_to_get.block(0, n, n + 1, 1) = row;

	DOCTEST_CHECK((m_to_get - ldl.reconstructed_matrix()).norm() <= eps);
}
