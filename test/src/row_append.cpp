#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>

#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using Scalar = double;
using namespace ldlt;

DOCTEST_TEST_CASE("row add") {
	isize n = 7;
	using T = f64;

	using Mat = ::Mat<T, colmajor>;
	using Vec = ::Vec<T>;

	Mat m = ldlt_test::rand::positive_definite_rand(n + 1, T(1e2));

	Mat l_target(n + 1, n + 1);
	Vec d_target(n + 1);

	Mat l_in(n, n);
	Vec d_in(n);

	Mat l_out(n + 1, n + 1);
	Vec d_out(n + 1);
	l_out.setZero();
	d_out.setZero();

	using LdltView = ldlt::LdltView<T>;
	using LdltViewMut = ldlt::LdltViewMut<T>;
	using MatrixView = ldlt::MatrixView<T, colmajor>;
	using MatrixViewMut = ldlt::MatrixViewMut<T, colmajor>;
	using VectorView = ldlt::VectorView<T>;
	using VectorViewMut = ldlt::VectorViewMut<T>;

	bool bool_values[] = {false, true};
	for (bool inplace : bool_values) {
		// factorize input matrix
		ldlt::factorize(
				LdltViewMut{
						{from_eigen, l_in},
						{from_eigen, d_in},
				},
				MatrixView{from_eigen, m}.block(0, 0, n, n));

		if (inplace) {
			l_out.topLeftCorner(n, n) = l_in.topLeftCorner(n, n);
			d_out.topRows(n) = d_in.topRows(n);
		}

		// append row
		ldlt::row_append(
				LdltViewMut{
						MatrixViewMut{from_eigen, l_out},
						VectorViewMut{from_eigen, d_out},
				},
				inplace //
						? (LdltView{
									MatrixView{from_eigen, l_out}.block(0, 0, n, n),
									VectorView{from_eigen, d_out}.segment(0, n),
							})
						: (LdltView{
									{from_eigen, l_in},
									{from_eigen, d_in},
							}),
				{from_eigen, Vec(m.row(n))});

		// compute target
		ldlt::factorize(
				LdltViewMut{
						{from_eigen, l_target},
						{from_eigen, d_target},
				},
				MatrixView{from_eigen, m});

		Scalar eps = Scalar(1e-10);

		DOCTEST_CHECK((l_target - l_out).norm() <= eps);
		DOCTEST_CHECK((d_target - d_out).norm() <= eps);
	}
}
