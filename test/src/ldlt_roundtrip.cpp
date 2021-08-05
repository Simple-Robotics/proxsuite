#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fmt/ostream.h>
#include <ldlt/ldlt.hpp>

using namespace ldlt;

template <typename T, Layout L>
using Mat = Eigen::
		Matrix<T, -1, -1, (L == colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <
		typename MatLhs,
		typename MatRhs,
		typename T = typename MatLhs::Scalar>
auto matmul(MatLhs const& a, MatRhs const& b) -> Mat<T, colmajor> {
	using Upscaled = typename std::
			conditional<std::is_floating_point<T>::value, long double, T>::type;

	return (Mat<T, colmajor>(a).template cast<Upscaled>().operator*(
							Mat<T, colmajor>(b).template cast<Upscaled>()))
	    .template cast<T>();
}

template <
		typename MatLhs,
		typename MatMid,
		typename MatRhs,
		typename T = typename MatLhs::Scalar>
auto matmul3(MatLhs const& a, MatMid const& b, MatRhs const& c)
		-> Mat<T, colmajor> {
	return ::matmul(::matmul(a, b), c);
}

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

	auto m_view = MatrixView<T, InL>{mat.data(), n};
	auto l_view = LowerTriangularMatrixViewMut<T, OutL>{l.data(), n};
	auto d_view = DiagonalMatrixViewMut<T>{d.data(), n};

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
void roundtrip_test(i32 min, i32 max, Fn ldlt_fn) {
	for (i32 n = min; n <= max; ++n) {

		auto data = generate_data<T, InL, OutL>(n);

		auto display = [&](fmt::string_view name, T err) {
			fmt::print("n = {}, {:<10}: {:>7.5e}\n", n, name, err);
		};
		fmt::print("{:-<40}\n", "");

		display("eigen", ::eigen_ldlt_roundtrip_error(data));
		display("ours", ::ldlt_roundtrip_error(data, ldlt_fn));
	}
}

auto main() -> int {
	roundtrip_test<f32, colmajor, colmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f32, rowmajor, colmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f32, colmajor, rowmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f32, rowmajor, rowmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);

	roundtrip_test<f64, colmajor, colmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f64, rowmajor, colmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f64, colmajor, rowmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
	roundtrip_test<f64, rowmajor, rowmajor>(
			1, 64, ldlt::factorize_defer_to_colmajor);
}
