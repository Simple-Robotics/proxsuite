#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using namespace ldlt;
using T = f64;

DOCTEST_TEST_CASE("row delete") {
	isize n = 7;

	using Mat = ::Mat<T, colmajor>;
	using Vec = ::Vec<T>;

	Mat m = ldlt_test::rand::positive_definite_rand(n, T(1e2));
	m.setRandom();
	m = m * m.transpose();

	Mat l_target(n - 1, n - 1);
	Vec d_target(n - 1);

	Mat l_in(n, n);
	Vec d_in(n);

	Mat l_out_storage(n, n);
	Vec d_out_storage(n);
	l_out_storage.setZero();
	d_out_storage.setZero();

	auto l_out = l_out_storage.topLeftCorner(n - 1, n - 1);
	auto d_out = d_out_storage.topRows(n - 1);

	using LdltView = ldlt::LdltView<T>;
	using LdltViewMut = ldlt::LdltViewMut<T>;

	bool bool_values[] = {false, true};
	for (bool inplace : bool_values) {
		for (isize idx = 0; idx < n; ++idx) {
			// factorize input matrix
			ldlt::factorize(
					LdltViewMut{
							{from_eigen, l_in},
							{from_eigen, d_in},
					},
					MatrixView<T, colmajor>{from_eigen, m});

			if (inplace) {
				l_out_storage = l_in;
				d_out_storage = d_in;
			}

			// delete ith row
			ldlt::row_delete(
					LdltViewMut{
							{from_eigen, l_out},
							{from_eigen, d_out},
					},
					inplace //
							? (LdltView{
										{from_eigen, l_out_storage},
										{from_eigen, d_out_storage},
								})
							: (LdltView{
										{from_eigen, l_in},
										{from_eigen, d_in},
								}),
					idx);

			// compute target
			{
				// delete idx'th row and column
				isize rem = n - idx - 1;
				l_target.topLeftCorner(idx, idx) = m.topLeftCorner(idx, idx);
				l_target.bottomLeftCorner(rem, idx) = m.bottomLeftCorner(rem, idx);

				l_target.topRightCorner(idx, rem) = m.topRightCorner(idx, rem);
				l_target.bottomRightCorner(rem, rem) = m.bottomRightCorner(rem, rem);

				// factorize matrix inplace
				ldlt::factorize(
						LdltViewMut{
								{from_eigen, l_target},
								{from_eigen, d_target},
						},
						MatrixView<T, colmajor>{from_eigen, l_target});
			}

			T eps = T(1e-10);
			DOCTEST_CHECK((l_target - l_out).norm() <= eps);
			DOCTEST_CHECK((d_target - d_out).norm() <= eps);
		}
	}
}
