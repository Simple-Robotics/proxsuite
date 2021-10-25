#include <doctest.h>
#include <util.hpp>
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>
#include <Eigen/Cholesky>

using namespace ldlt;

template <typename T>
struct Data {
	Mat<T, colmajor> mat;
	Mat<T, colmajor> w;
	Vec<T> alpha;
	Mat<T, colmajor> l;
	Vec<T> d;
};

template <typename T>
auto generate_data(isize n) -> Data<T> {
	ldlt_test::rand::set_seed(ldlt::u64(n));
	isize r = 7;
	Mat<T, colmajor> mat = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
	Mat<T, colmajor> w = ldlt_test::rand::matrix_rand<T>(n, r);
	Vec<T> alpha = ldlt_test::rand::vector_rand<T>(r);

	Mat<T, colmajor> l(n, n);
	l.setZero();

	Vec<T> d(n);
	d.setZero();

	return {
			LDLT_FWD(mat),
			LDLT_FWD(w),
			LDLT_FWD(alpha),
			LDLT_FWD(l),
			LDLT_FWD(d),
	};
}

template <typename T>
auto ldlt_roundtrip_error(Data<T>& data) -> T {
	auto const& mat = data.mat;
	auto const& w = data.w;
	auto const& alpha = data.alpha;
	auto& l = data.l;
	auto& d = data.d;
	isize n = w.rows();
	isize r = w.cols();

	l.setZero();
	d.setZero();

	auto m_view = MatrixView<T, colmajor>{from_eigen, mat};
	auto ldl_view = LdltViewMut<T>{
			{from_eigen, l},
			{from_eigen, d},
	};

	{
		LDLT_MULTI_WORKSPACE_MEMORY(
				(z, Uninit, Mat(n, r), LDLT_CACHELINE_BYTES, T),
				(a, Uninit, Vec(r), LDLT_CACHELINE_BYTES, T));
		z.to_eigen() = w;
		a.to_eigen() = alpha;

		EigenNoAlloc _{};
		factorize(ldl_view, m_view);
		detail::rank_r_update( //
				ldl_view,
				ldl_view.as_const(),
				z,
				a);
	}

	return (matmul3(l, d.asDiagonal(), l.transpose()) -
	        (mat + matmul3(w, alpha.asDiagonal(), w.transpose())))
	    .norm();
}

template <typename T>
auto eigen_ldlt_roundtrip_error(Data<T>& data) -> T {
	auto const& mat = data.mat;
	auto const& w = data.w;
	auto const& alpha = data.alpha;
	auto ldlt = mat.ldlt();
	for (isize i = 0; i < data.w.cols(); ++i) {
		ldlt.rankUpdate(data.w.col(i), data.alpha(i));
	}
	auto const& L = ldlt.matrixL();
	auto const& P = ldlt.transpositionsP();
	auto const& D = ldlt.vectorD();

	Mat<T, colmajor> tmp = P.transpose() * Mat<T, colmajor>(L);
	return (matmul3(tmp, D.asDiagonal(), tmp.transpose()) -
	        (mat + matmul3(w, alpha.asDiagonal(), w.transpose())))
	    .norm();
}

template <typename T>
auto roundtrip_test(isize n) -> T {
	auto data = generate_data<T>(n);

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

DOCTEST_TEST_CASE("rank one update: roundtrip") {
	using T = f64;
	isize min = 1;
	isize max = 64;

	for (isize i = min; i <= max; ++i) {
		DOCTEST_CHECK(roundtrip_test<T>(i) <= T(1e3));
	}
}
