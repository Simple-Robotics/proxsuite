#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using Scalar = double;

DOCTEST_TEST_CASE_TEMPLATE(
		"row add",
		L,
		ldlt::detail::constant<ldlt::Layout, ldlt::colmajor>,
		ldlt::detail::constant<ldlt::Layout, ldlt::rowmajor>) {
	ldlt::i32 n = 7;

	using Mat = Eigen::Matrix<
			Scalar,
			Eigen::Dynamic,
			Eigen::Dynamic,
			L::value == ldlt::colmajor ? Eigen::ColMajor : Eigen::RowMajor>;
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	Mat m(n + 1, n + 1);
	m.setRandom();
	m = m * m.transpose();

	Mat l_target(n + 1, n + 1);
	Vec d_target(n + 1);

	Mat l_in(n, n);
	Vec d_in(n);

	Mat l_out(n + 1, n + 1);
	Vec d_out(n + 1);
	l_out.setZero();
	d_out.setZero();

	using LdltView = ldlt::LdltView<Scalar, L::value>;
	using LdltViewMut = ldlt::LdltViewMut<Scalar, L::value>;

	bool bool_values[] = {false, true};
	for (bool inplace : bool_values) {
		// factorize input matrix
		ldlt::factorize(
				LdltViewMut{
						ldlt::detail::from_eigen_matrix_mut(l_in),
						ldlt::detail::from_eigen_vector_mut(d_in),
				},
				ldlt::detail::from_eigen_matrix(m.topLeftCorner(n, n)));

		if (inplace) {
			l_out.topLeftCorner(n, n) = l_in.topLeftCorner(n, n);
			d_out.topRows(n) = d_in.topRows(n);
		}

		// append row
		ldlt::row_append(
				LdltViewMut{
						ldlt::detail::from_eigen_matrix_mut(l_out).block(0, 0, n, n),
						ldlt::detail::from_eigen_vector_mut(d_out).segment(0, n),
				},
				inplace //
						? (LdltView{
									ldlt::detail::from_eigen_matrix(l_out).block(0, 0, n, n),
									ldlt::detail::from_eigen_vector(d_out).segment(0, n),
							})
						: (LdltView{
									ldlt::detail::from_eigen_matrix(l_in),
									ldlt::detail::from_eigen_vector(d_in),
							}),
				ldlt::detail::from_eigen_vector(Vec(m.row(n))));

		// compute target
		ldlt::factorize(
				LdltViewMut{
						ldlt::detail::from_eigen_matrix_mut(l_target),
						ldlt::detail::from_eigen_vector_mut(d_target),
				},
				ldlt::detail::from_eigen_matrix(m));

		Scalar eps = Scalar(1e-10);

		DOCTEST_CHECK((l_target - l_out).norm() <= eps);
		DOCTEST_CHECK((d_target - d_out).norm() <= eps);
	}
}
