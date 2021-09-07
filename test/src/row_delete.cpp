#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using Scalar = double;

DOCTEST_TEST_CASE_TEMPLATE(
		"row delete", L, ldlt::detail::constant<ldlt::Layout, ldlt::colmajor>
		// ,ldlt::detail::constant<ldlt::Layout, ldlt::rowmajor>
    // requires simde store_scatter
) {
	ldlt::i32 n = 7;

	using Mat = Eigen::Matrix<
			Scalar,
			Eigen::Dynamic,
			Eigen::Dynamic,
			L::value == ldlt::colmajor ? Eigen::ColMajor : Eigen::RowMajor>;
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	Mat m(n, n);
	m.setRandom();
	m = m * m.transpose();

	Mat l_target(n - 1, n - 1);
	Vec d_target(n - 1);

	Mat l_in(n, n);
	Vec d_in(n);

	Mat l_out(n - 1, n - 1);
	Vec d_out(n - 1);
	l_out.setZero();
	d_out.setZero();

	using LdltView = ldlt::LdltView<Scalar, L::value>;
	using LdltViewMut = ldlt::LdltViewMut<Scalar, L::value>;

	bool bool_values[] = {false, true};
	for (bool inplace : bool_values) {
		for (ldlt::i32 idx = 0; idx < n; ++idx) {
			// factorize input matrix
			ldlt::factorize(
					LdltViewMut{
							ldlt::detail::from_eigen_matrix_mut(l_in),
							ldlt::detail::from_eigen_vector_mut(d_in),
					},
					ldlt::detail::from_eigen_matrix(m));

			// delete ith row
			ldlt::row_delete(
					inplace //
							? (LdltViewMut{
										ldlt::detail::from_eigen_matrix_mut(l_in).block(
												0, 0, n - 1, n - 1),
										ldlt::detail::from_eigen_vector_mut(d_in).segment(0, n - 1),
								})
							: (LdltViewMut{
										ldlt::detail::from_eigen_matrix_mut(l_out),
										ldlt::detail::from_eigen_vector_mut(d_out),
								}),
					LdltView{
							ldlt::detail::from_eigen_matrix(l_in),
							ldlt::detail::from_eigen_vector(d_in),
					},
					idx);

			// compute target
			{
				// delete idx'th row and column
				ldlt::i32 rem = n - idx - 1;
				l_target.topLeftCorner(idx, idx) = m.topLeftCorner(idx, idx);
				l_target.bottomLeftCorner(rem, idx) = m.bottomLeftCorner(rem, idx);

				l_target.topRightCorner(idx, rem) = m.topRightCorner(idx, rem);
				l_target.bottomRightCorner(rem, rem) = m.bottomRightCorner(rem, rem);

				// factorize matrix inplace
				ldlt::factorize(
						LdltViewMut{
								ldlt::detail::from_eigen_matrix_mut(l_target),
								ldlt::detail::from_eigen_vector_mut(d_target),
						},
						ldlt::detail::from_eigen_matrix(l_target));
			}

			Scalar eps = Scalar(1e-10);
			if (inplace) {
				DOCTEST_CHECK(
						(l_target - l_in.topLeftCorner(n - 1, n - 1)).norm() <= eps);
				DOCTEST_CHECK((d_target - d_in.topRows(n - 1)).norm() <= eps);
			} else {
				DOCTEST_CHECK((l_target - l_out).norm() <= eps);
				DOCTEST_CHECK((d_target - d_out).norm() <= eps);
			}
		}
	}
}
