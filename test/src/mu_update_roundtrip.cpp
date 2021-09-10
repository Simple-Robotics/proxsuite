#include <doctest.h>
#include <util.hpp>
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>
#include <Eigen/Cholesky>

using namespace ldlt;

template <typename T>
struct Data {
	Mat<T, colmajor> mat;
	Vec<T> diag_diff;
	isize index;
	Mat<T, colmajor> l;
	Vec<T> d;
};

template <typename T>
auto generate_data(isize n) -> Data<T> {
	isize index = n / 2;
	isize max_n_eq = 2;
	isize n_eq = ((max_n_eq + index) < n) ? max_n_eq : n - index;
	ldlt_test::rand::set_seed(u64(n));
	Mat<T, colmajor> mat = ldlt_test::rand::positive_definite_rand<T>(n, T(1e2));
	Vec<T> diag_diff = ldlt_test::rand::vector_rand<T>(n_eq);

	Mat<T, colmajor> l(n, n);
	Vec<T> d(n);

	return {
			LDLT_FWD(mat),
			LDLT_FWD(diag_diff),
			index,
			LDLT_FWD(l),
			LDLT_FWD(d),
	};
}

template <typename T>
auto ldlt_roundtrip_error(Data<T>& data) -> T {
	auto const& mat = data.mat;
	auto const& diag_diff = data.diag_diff;
	auto& l = data.l;
	auto& d = data.d;

	l.setZero();
	d.setZero();

	auto m_view = MatrixView<T, colmajor>{from_eigen, mat};
	auto ldl_view = LdltViewMut<T>{
			{from_eigen, l},
			{from_eigen, d},
	};

	{
		EigenNoAlloc _{};
		factorize(ldl_view, m_view);
		diagonal_update( //
				ldl_view,
				ldl_view.as_const(),
				{from_eigen, diag_diff},
				data.index);
	}

	auto new_mat = mat;
	for (isize i = 0; i < diag_diff.rows(); ++i) {
		auto ii = data.index + i;
		new_mat(ii, ii) += diag_diff(i);
	}

	return (matmul3(l, d.asDiagonal(), l.transpose()) - new_mat).norm();
}

template <typename T>
auto roundtrip_test(isize n) -> T {
	auto data = generate_data<T>(n);
	return ::ldlt_roundtrip_error(data);
}

DOCTEST_TEST_CASE("rank one update: roundtrip") {
	isize min = 1;
	isize max = 64;
	using Scalar = f64;

	for (isize i = min; i <= max; ++i) {
		DOCTEST_CHECK(
				roundtrip_test<Scalar>(i) <=
				std::numeric_limits<Scalar>::epsilon() * Scalar(1e3));
	}
}
