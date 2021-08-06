#include <doctest.h>
#include <util.hpp>
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>
#include <Eigen/Cholesky>

using namespace ldlt;

template <typename T, Layout InL, Layout OutL>
struct Data {
	Mat<T, InL> mat;
	Vec<T> w;
	T alpha;
	Mat<T, OutL> l;
	Vec<T> d;
};

template <typename T, Layout InL, Layout OutL>
auto generate_data(i32 n) -> Data<T, InL, OutL> {
	Mat<T, InL> mat(n, n);
	Vec<T> w(n);
	T alpha;

	Mat<T, OutL> l(n, n);
	Vec<T> d(n);
	std::srand(unsigned(n));
	mat.setRandom();
	{
		Vec<T> _alpha(1);
		_alpha.setRandom();
		alpha = _alpha(0);
	}
	w.setRandom();
	mat = (mat.transpose() * mat).eval();

	return {
			LDLT_FWD(mat),
			LDLT_FWD(w),
			LDLT_FWD(alpha),
			LDLT_FWD(l),
			LDLT_FWD(d),
	};
}

template <typename T, Layout InL, Layout OutL>
auto ldlt_roundtrip_error(Data<T, InL, OutL>& data) -> T {
	auto const& mat = data.mat;
	auto const& w = data.w;
	auto const& alpha = data.alpha;
	auto& l = data.l;
	auto& d = data.d;

	l.setZero();
	d.setZero();
	i32 n = i32(mat.rows());

	auto m_view = MatrixView<T, InL>{mat.data(), n, n};
	auto l_view = MatrixViewMut<T, OutL>{l.data(), n, n};
	auto d_view = VectorViewMut<T>{d.data(), n};
	auto w_view = VectorView<T>{w.data(), n};

	factorize(l_view, d_view, m_view);
	detail::rank1_update(l_view, d_view, w_view, alpha);

	return (matmul3(l, d.asDiagonal(), l.transpose()) -
	        (mat + alpha * w * w.transpose()))
	    .norm();
}

template <typename T, Layout InL, Layout OutL>
auto eigen_ldlt_roundtrip_error(Data<T, InL, OutL>& data) -> T {
	auto const& mat = data.mat;
	auto const& w = data.w;
	auto const& alpha = data.alpha;
	auto ldlt = mat.ldlt();
	ldlt.rankUpdate(data.w, data.alpha);
	auto const& L = ldlt.matrixL();
	auto const& P = ldlt.transpositionsP();
	auto const& D = ldlt.vectorD();

	Mat<T, colmajor> tmp = P.transpose() * Mat<T, colmajor>(L);
	return (matmul3(tmp, D.asDiagonal(), tmp.transpose()) -
	        (mat + alpha * w * w.transpose()))
	    .norm();
}

template <typename T, Layout InL, Layout OutL>
auto roundtrip_test(i32 n) -> T {
	auto data = generate_data<T, InL, OutL>(n);

	T err_eigen = ::eigen_ldlt_roundtrip_error(data);
	T err_ours = ::ldlt_roundtrip_error(data);
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
		"rank one update: roundtrip", Args, detail::type_sequence<f64, C, C>) {
	i32 min = 1;
	i32 max = 128;
	using Scalar = detail::typeseq_ith<0, Args>;
	constexpr auto InL = detail::typeseq_ith<1, Args>::value;
	constexpr auto OutL = detail::typeseq_ith<2, Args>::value;

	for (i32 i = min; i <= max; ++i) {
		DOCTEST_CHECK(roundtrip_test<Scalar, InL, OutL>(i) <= Scalar(1e4));
	}
}
