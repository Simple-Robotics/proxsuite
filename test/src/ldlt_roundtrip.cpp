#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fmt/ostream.h>
#include <ldlt/ldlt.hpp>

using namespace ldlt;

template <typename T>
struct DoNotDeduceImpl {
	using Type = T;
};

template <typename T>
using DoNotDeduce = typename DoNotDeduceImpl<T>::Type;

template <typename T>
using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, -1, 1>;

template <typename T>
auto matmul(Mat<T> const& a, DoNotDeduce<Mat<T>> const& b) -> Mat<T> {
	using Upscaled = typename std::
			conditional<std::is_floating_point<T>::value, long double, T>::type;

	return (a.template cast<Upscaled>() * b.template cast<Upscaled>())
	    .template cast<T>();
}

template <typename T>
auto matmul3(
		Mat<T> const& a, DoNotDeduce<Mat<T>> const& b, DoNotDeduce<Mat<T>> const& c)
		-> Mat<T> {
	return ::matmul(::matmul(a, b), c);
}

template <typename T>
struct Data {
	Mat<T> mat;
	Mat<T> l;
	Vec<T> d;
};

template <typename T>
auto generate_data(i32 n) -> Data<T> {
	Mat<T> mat(n, n);
	Mat<T> l(n, n);
	Vec<T> d(n);
	std::srand(unsigned(n));
	mat.setRandom();
	mat = (mat.transpose() * mat).eval();

	return {LDLT_FWD(mat), LDLT_FWD(l), LDLT_FWD(d)};
}

template <typename T>
auto ldlt_roundtrip_error(Data<T>& data) -> T {
	auto const& mat = data.mat;
	auto& l = data.l;
	auto& d = data.d;
	l.setZero();
	d.setZero();
	i32 n = i32(mat.rows());

	auto m_view = MatrixView<T, colmajor>{mat.data(), n};
	auto l_view = LowerTriangularMatrixViewMut<T, colmajor>{l.data(), n};
	auto d_view = DiagonalMatrixViewMut<T>{d.data(), n};

	ldlt::factorize_ldlt_unblocked(l_view, d_view, m_view);

	return (matmul3(l, d.asDiagonal(), l.transpose()) - mat).norm();
}

template <typename T>
auto eigen_ldlt_roundtrip_error(Data<T>& data) -> T {
	auto ldlt = data.mat.ldlt();
	auto const& L = ldlt.matrixL();
	auto const& P = ldlt.transpositionsP();
	auto const& D = ldlt.vectorD();

	Mat<T> tmp = P.transpose() * Mat<T>(L);
	return (matmul3(tmp, D.asDiagonal(), tmp.transpose()) - data.mat).norm();
}

template <typename T>
void roundtrip_test(i32 min, i32 max) {
	for (i32 n = min; n <= max; ++n) {

		auto data = generate_data<T>(n);

		auto display = [&](fmt::string_view name, T err) {
			fmt::print("n = {}, {:<10}: {:>7.5e}\n", n, name, err);
		};
		fmt::print("{:-<40}\n", "");

		display("eigen", ::eigen_ldlt_roundtrip_error(data));
		display("ours", ::ldlt_roundtrip_error(data));
	}
}

auto main() -> int {
	roundtrip_test<f32>(1, 64);
	roundtrip_test<f64>(1, 64);
}
