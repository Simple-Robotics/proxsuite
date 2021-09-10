#include <Eigen/Cholesky>
#include <fmt/ostream.h>
#include <ldlt/factorize.hpp>
#include <util.hpp>
#include <limits>
#include <doctest.h>

using namespace ldlt;

template <typename T, Layout L>
struct Data {
	Mat<T, L> mat;
	Mat<T, colmajor> l;
	Vec<T> d;
};

template <typename T, Layout L>
auto generate_data(isize n) -> Data<T, L> {
	ldlt_test::rand::set_seed(u64(n));
	Mat<T, L> mat = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
	Mat<T, colmajor> l(n, n);
	Vec<T> d(n);
	return {LDLT_FWD(mat), LDLT_FWD(l), LDLT_FWD(d)};
}

template <typename T, Layout L, typename S>
auto ldlt_roundtrip_error(Data<T, L>& data, S strategy) -> T {
	auto const& mat = data.mat;
	auto& l = data.l;
	auto& d = data.d;
	l.setZero();
	d.setZero();

	auto m_view = MatrixView<T, L>{from_eigen, mat};
	auto ldl_view = LdltViewMut<T>{
			{from_eigen, l},
			{from_eigen, d},
	};

	{
		EigenNoAlloc _{};
		factorize(ldl_view, m_view, strategy);
	}

	return (matmul3(l, d.asDiagonal(), l.transpose()) - mat).norm();
}

template <typename T, Layout L>
auto eigen_ldlt_roundtrip_error(Data<T, L>& data) -> T {
	auto ldlt = data.mat.ldlt();
	auto const& l = ldlt.matrixL();
	auto const& p = ldlt.transpositionsP();
	auto const& d = ldlt.vectorD();

	Mat<T, colmajor> tmp = p.transpose() * Mat<T, colmajor>(l);
	return (matmul3(tmp, d.asDiagonal(), tmp.transpose()) - data.mat).norm();
}

template <typename T, Layout L, typename S>
auto roundtrip_test(isize n, S strategy) -> T {
	auto data = generate_data<T, L>(n);

	T err_eigen = ::eigen_ldlt_roundtrip_error(data);
	T err_ours = ::ldlt_roundtrip_error(data, strategy);
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
		detail::type_sequence<f32, C, factorization_strategy::Standard>,
		detail::type_sequence<f32, R, factorization_strategy::Standard>) {
	isize min = 1;
	isize max = 64;
	using Scalar = detail::typeseq_ith<0, Args>;
	constexpr auto InL = detail::typeseq_ith<1, Args>::value;
	using Strategy = detail::typeseq_ith<2, Args>;

	for (isize i = min; i <= max; ++i) {
		DOCTEST_CHECK(roundtrip_test<Scalar, InL>(i, Strategy{}) <= Scalar(10));
	}
}
