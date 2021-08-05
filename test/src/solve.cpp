#include <util.hpp>
#include <doctest.h>
#include <Eigen/Cholesky>
#include <ldlt/solve.hpp>
#include <ldlt/factorize.hpp>
#include <fmt/ostream.h>

using namespace ldlt;

using C = detail::constant<Layout, colmajor>;
using R = detail::constant<Layout, rowmajor>;

DOCTEST_TEST_CASE_TEMPLATE("solve", LType, C, R) {
	using T = double;

	constexpr Layout L = LType::value;
	for (i32 i = 1; i <= 128; ++i) {
		std::srand(unsigned(i));
		Mat<T, colmajor> a(i, i);
		Mat<T, L> l(i, i);
		Vec<T> d(i);

		Vec<T> b(i);
		Vec<T> x(i);
		Vec<T> x_eigen(i);
		Vec<long double> x_eigen_upscaled(i);

		a.setRandom();
		a = a.transpose() * a;
		b.setRandom();

		auto a_view = ldlt::MatrixView<T, colmajor>{a.data(), i, i};
		auto l_view = ldlt::MatrixViewMut<T, L>{l.data(), i, i};
		auto d_view = ldlt::VectorViewMut<T>{d.data(), i};

		auto x_view = ldlt::VectorViewMut<T>{x.data(), i};
		auto b_view = ldlt::VectorView<T>{b.data(), i};

		factorize(l_view, d_view, a_view);
		solve(x_view, l_view.as_const(), d_view.as_const(), b_view);
		x_eigen = a.ldlt().solve(b);
		x_eigen_upscaled =
				a.cast<long double>().ldlt().solve(b.cast<long double>());

		auto err_ours = T((x.cast<long double>() - x_eigen_upscaled).norm());
		auto err_eigen = T((x_eigen.cast<long double>() - x_eigen_upscaled).norm());

		if (err_ours > 0) {
			DOCTEST_CHECK(err_ours / err_eigen < T(1e3));
		}
	}
}
