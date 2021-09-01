#include <doctest.h>
#include <util.hpp>
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>
#include <Eigen/Cholesky>

using namespace ldlt;

template <typename T, Layout InL, Layout OutL>
struct Data {
	Mat<T, InL> mat;
	Vec<T> diag_diff;
	i32 index;
	Mat<T, OutL> l;
	Vec<T> d;
};

template <typename T, Layout InL, Layout OutL>
auto generate_data(i32 n) -> Data<T, InL, OutL> {
	i32 index = n / 2;
	i32 max_n_eq = 2;
	i32 n_eq = ((max_n_eq + index) < n) ? max_n_eq : n - index;
	Mat<T, InL> mat(n, n);
	Vec<T> diag_diff(n_eq);

	Mat<T, OutL> l(n, n);
	Vec<T> d(n);
	std::srand(unsigned(n));
	mat.setRandom();
	diag_diff.setRandom();
	mat = (mat.transpose() * mat).eval();

	return {
			LDLT_FWD(mat),
			LDLT_FWD(diag_diff),
			index,
			LDLT_FWD(l),
			LDLT_FWD(d),
	};
}

template <typename T, Layout InL, Layout OutL>
auto ldlt_roundtrip_error(Data<T, InL, OutL>& data) -> T {
	auto const& mat = data.mat;
	auto const& diag_diff = data.diag_diff;
	auto& l = data.l;
	auto& d = data.d;

	l.setZero();
	d.setZero();

	auto m_view = detail::from_eigen_matrix(mat);
	auto ldl_view = LdltViewMut<T, OutL>{
			detail::from_eigen_matrix_mut(l),
			detail::from_eigen_vector_mut(d),
	};

	{
		EigenNoAlloc _{};
		factorize(ldl_view, m_view);
		diagonal_update( //
				ldl_view,
				ldl_view.as_const(),
				detail::from_eigen_vector(diag_diff),
				data.index);
	}

	auto new_mat = mat;
	for (i32 i = 0; i < diag_diff.rows(); ++i) {
		auto ii = data.index + i;
		new_mat(ii, ii) += diag_diff(i);
	}

	return (matmul3(l, d.asDiagonal(), l.transpose()) - new_mat).norm();
}

template <typename T, Layout InL, Layout OutL>
auto roundtrip_test(i32 n) -> T {
	auto data = generate_data<T, InL, OutL>(n);
	return ::ldlt_roundtrip_error(data);
}

using C = detail::constant<Layout, colmajor>;
using R = detail::constant<Layout, rowmajor>;

DOCTEST_TEST_CASE_TEMPLATE(
		"rank one update: roundtrip", Args, detail::type_sequence<f64, C, C>) {
	i32 min = 1;
	i32 max = 64;
	using Scalar = detail::typeseq_ith<0, Args>;
	constexpr auto InL = detail::typeseq_ith<1, Args>::value;
	constexpr auto OutL = detail::typeseq_ith<2, Args>::value;

	for (i32 i = min; i <= max; ++i) {
		DOCTEST_CHECK(
				roundtrip_test<Scalar, InL, OutL>(i) <=
				std::numeric_limits<Scalar>::epsilon() * Scalar(1e3));
	}
}
