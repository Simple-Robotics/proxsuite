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

DOCTEST_TEST_CASE("append") {

	using namespace ldlt::tags;

	isize dim = 3;
	T eps = T(1e-10);
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_m_init, Init, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_m_to_get_, Init, Mat(dim + 1, dim + 1), LDLT_CACHELINE_BYTES, T),
			(row_, Init, Vec(dim + 1), LDLT_CACHELINE_BYTES, T));

	auto m_init = _m_init.to_eigen();
	auto m_to_get = _m_to_get_.to_eigen();
	auto col = row_.to_eigen();

	m_init.diagonal().setRandom();
	col.setRandom();

	m_to_get.topLeftCorner(dim, dim) = m_init;
	m_to_get.col(dim) = col;
	m_to_get.row(dim) = col.transpose();

	auto ldl = ldlt::Ldlt<T>{decompose, m_init};
	ldl.insert_at(dim, col);

	DOCTEST_CHECK((m_to_get - ldl.reconstructed_matrix()).norm() <= eps);
}

DOCTEST_TEST_CASE("append Maros example HS118") {

	using namespace ldlt::tags;

	isize dim = 15;
	T eps = T(1e-10);

	LDLT_MULTI_WORKSPACE_MEMORY(
			(_m_init, Init, Mat(dim, dim), LDLT_CACHELINE_BYTES, T),
			(_m_to_get_, Init, Mat(dim + 1, dim + 1), LDLT_CACHELINE_BYTES, T),
			(row_, Init, Vec(dim + 1), LDLT_CACHELINE_BYTES, T));

	auto m_init = _m_init.to_eigen();
	auto m_to_get = _m_to_get_.to_eigen();
	auto row = row_.to_eigen();

	m_init.diagonal() << 0.000201, 0.000201, 0.000301, 0.000201, 0.000201,
			0.000301, 0.000201, 0.000201, 0.000301, 0.000201, 0.000201, 0.000301,
			0.000201, 0.000201, 0.000301;

	ldlt::Ldlt<T> ldl{decompose, m_init};

	DOCTEST_CHECK((m_init - ldl.reconstructed_matrix()).norm() <= eps);

	row << 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1;

	ldl.insert_at(dim, row);

	m_to_get.diagonal() << 0.000201, 0.000201, 0.000301, 0.000201, 0.000201,
			0.000301, 0.000201, 0.000201, 0.000301, 0.000201, 0.000201, 0.000301,
			0.000201, 0.000201, 0.000301, -0.1;

	m_to_get.block(dim, 0, 1, dim + 1) = row.transpose(); // insert at the end
	m_to_get.block(0, dim, dim + 1, 1) = row;

	DOCTEST_CHECK((m_to_get - ldl.reconstructed_matrix()).norm() <= eps);
}
