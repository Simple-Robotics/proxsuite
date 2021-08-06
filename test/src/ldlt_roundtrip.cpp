#include <Eigen/Cholesky>
#include <fmt/ostream.h>
#include <ldlt/factorize.hpp>
#include <util.hpp>
#include <limits>
#include <doctest.h>

using namespace ldlt;

template <typename T, Layout InL, Layout OutL>
struct Data {
	Mat<T, InL> mat;
	Mat<T, OutL> l;
	Vec<T> d;
};

template <typename T, Layout InL, Layout OutL>
auto generate_data(i32 n) -> Data<T, InL, OutL> {
	Mat<T, InL> mat(n, n);
	Mat<T, OutL> l(n, n);
	Vec<T> d(n);
	std::srand(unsigned(n));
	mat.setRandom();
	mat = (mat.transpose() * mat).eval();

	return {LDLT_FWD(mat), LDLT_FWD(l), LDLT_FWD(d)};
}

template <typename T, Layout InL, Layout OutL, typename Fn>
auto ldlt_roundtrip_error(Data<T, InL, OutL>& data, Fn ldlt_fn) -> T {
	auto const& mat = data.mat;
	auto& l = data.l;
	auto& d = data.d;
	l.setZero();
	d.setZero();
	i32 n = i32(mat.rows());

	auto m_view = MatrixView<T, InL>{mat.data(), n, n};
	auto l_view = MatrixViewMut<T, OutL>{l.data(), n, n};
	auto d_view = VectorViewMut<T>{d.data(), n};

	ldlt_fn(l_view, d_view, m_view);

	return (matmul3(l, d.asDiagonal(), l.transpose()) - mat).norm();
}

template <typename T, Layout InL, Layout OutL>
auto eigen_ldlt_roundtrip_error(Data<T, InL, OutL>& data) -> T {
	auto ldlt = data.mat.ldlt();
	auto const& L = ldlt.matrixL();
	auto const& P = ldlt.transpositionsP();
	auto const& D = ldlt.vectorD();

	Mat<T, colmajor> tmp = P.transpose() * Mat<T, colmajor>(L);
	return (matmul3(tmp, D.asDiagonal(), tmp.transpose()) - data.mat).norm();
}

template <typename T, Layout InL, Layout OutL, typename Fn>
auto roundtrip_test(i32 n, Fn ldlt_fn) -> T {
	auto data = generate_data<T, InL, OutL>(n);

	T err_eigen = ::eigen_ldlt_roundtrip_error(data);
	T err_ours = ::ldlt_roundtrip_error(data, ldlt_fn);
	if (err_ours == 0) {
		return T(0);
	}
	if (err_eigen == 0) {
		err_eigen = std::numeric_limits<T>::epsilon();
	}

	return err_ours / err_eigen;
}

using C = detail::constant<Layout, colmajor>;
using R = detail::constant<Layout, rowmajor>;

DOCTEST_TEST_CASE_TEMPLATE(
		"factorize: roundtrip",
		Args,
		detail::type_sequence<f32, C, C, nb::factorize>,
		detail::type_sequence<f32, C, C, nb::factorize_defer_to_colmajor>,
		detail::type_sequence<f32, R, C, nb::factorize>,
		detail::type_sequence<f32, R, C, nb::factorize_defer_to_colmajor>,
		detail::type_sequence<f32, C, R, nb::factorize>,
		detail::type_sequence<f32, C, R, nb::factorize_defer_to_colmajor>,
		detail::type_sequence<f32, R, R, nb::factorize>,
		detail::type_sequence<f32, R, R, nb::factorize_defer_to_colmajor>) {
	i32 min = 1;
	i32 max = 128;
	using Scalar = detail::typeseq_ith<0, Args>;
	constexpr auto InL = detail::typeseq_ith<1, Args>::value;
	constexpr auto OutL = detail::typeseq_ith<2, Args>::value;
	using Ldlt = detail::typeseq_ith<3, Args>;

	for (i32 i = min; i <= max; ++i) {
		DOCTEST_CHECK(roundtrip_test<Scalar, InL, OutL>(i, Ldlt{}) <= Scalar(10));
	}
}
