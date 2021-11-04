// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at
// http://www.opensource.org/licenses/mit-license.php

#ifndef INRIA_LDLT_CNPY_HPP_TAEAIC6RS
#define INRIA_LDLT_CNPY_HPP_TAEAIC6RS

#include <memory>
#include <cstring>
#include <complex>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <memory>
#include <cstdint>
#include <Eigen/Core>

#define CNPY_FWD(x) static_cast<decltype(x)&&>(x)
#define CNPY_ASSERT(Cond)                                                      \
	((Cond) ? (void)0                                                            \
	        : ::cnpy::detail::terminate_with_message(                            \
								"assertion failed: " #Cond,                                    \
								sizeof("assertion failed: " #Cond) - 1))

#define CNPY_LITERAL(X) X, sizeof(X) - 1

namespace cnpy {
namespace detail {

enum struct LoadVecResult : int {
	success = 0,
	failed_file = 1,
	failed_dtype = 2,
	failed_ndim = 3,
};

enum struct LoadMatResult : int {
	success = 0,
	success_transpose = -1,

	failed_file = 1,
	failed_dtype = 2,
	failed_ndim = 3,
};

void terminate_with_message(char const* msg, size_t len);

template <typename T>
struct TypeCode;
template <>
struct TypeCode<float> {
	static constexpr char value = 'f';
};
template <>
struct TypeCode<double> {
	static constexpr char value = 'f';
};

struct FromByteRepr {};
struct FromLiteral {};
struct FromRawParts {};
struct FromRange {};

struct StrView {
	char const* data;
	size_t size;

	StrView(FromRawParts /*tag*/, char const* data_, size_t size_) noexcept
			: data{data_}, size{size_} {}

	template <typename T>
	StrView(FromRange /*tag*/, T const& str) noexcept
			: data{str.data()}, size{str.size()} {}

	template <typename T>
	StrView(FromByteRepr /*tag*/, T const& val) noexcept
			: data{&reinterpret_cast<char const&>(val)}, size{sizeof(T)} {}

	template <size_t N>
	constexpr StrView(FromLiteral /*tag*/, char const (&literal)[N]) noexcept
			: data{literal}, size{N - 1} {}
};

} // namespace detail

namespace detail {
inline auto BigEndianTest() -> char {
	int x = 1;
	char buf[sizeof(x)];
	std::memcpy(&buf, &x, sizeof(x));
	return buf[0] != 0 ? '<' : '>';
}
template <typename T>
auto create_npy_header(std::vector<size_t> const& shape) -> std::vector<char>;

auto create_npy_header(
		std::vector<size_t> const& shape, size_t sizeof_T, char type_code)
		-> std::vector<char>;
void parse_npy_header(
		FILE* fp,
		size_t& word_size,
		std::vector<size_t>& shape,
		bool& fortran_order);
void parse_npy_header(
		unsigned char* buffer,
		size_t& word_size,
		std::vector<size_t>& shape,
		bool& fortran_order);

void npy_vsave(
		char const* fname,
		void const* vdata,
		size_t sizeof_T,
		char type_code,
		size_t const* shape,
		size_t ndim,
		char const* mode);

auto npy_vload_vec(
		std::string const& fname,
		size_t sizeof_T,
		void* vec,
		void* (*ptr)(void*),
		void (*resize)(void*, size_t rows)) -> LoadVecResult;
auto npy_vload_mat(
		std::string const& fname,
		size_t sizeof_T,
		void* vec,
		void* (*ptr)(void*),
		void (*resize)(void*, size_t rows, size_t cols)) -> LoadMatResult;
} // namespace detail

template <typename D>
void npy_save_mat(
		std::string const& fname,
		Eigen::MatrixBase<D> const& mat,
		std::string const& mode = "w") {

	auto const& eval = mat.eval();
	size_t rowcol[] = {eval.rows(), eval.cols()};

	using T = typename D::Scalar;

	detail::npy_vsave( //
			fname.c_str(),
			eval.data(),
			sizeof(T),
			detail::TypeCode<T>::value,
			&rowcol[0],
			2,
			mode.c_str());
}

template <typename D>
void npy_save_vec(
		std::string const& fname,
		Eigen::MatrixBase<D> const& vec,
		const std::string& mode = "w") {

	auto const& eval = vec.eval();
	using T = typename D::Scalar;
	CNPY_ASSERT(vec.cols() == 1);
	std::size_t size = vec.rows();

	detail::npy_vsave( //
			fname.c_str(),
			vec.data(),
			sizeof(T),
			detail::TypeCode<T>::value,
			&size,
			1,
			mode.c_str());
}

template <typename T>
auto npy_load_vec(std::string const& fname)
		-> Eigen::Matrix<T, Eigen::Dynamic, 1> {
	using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	Vec out;

	using Res = detail::LoadVecResult;
	using detail::terminate_with_message;

	Res res = detail::npy_vload_vec( //
			fname.c_str(),
			sizeof(T),
			std::addressof(out),
			+[](void* vec) -> void* { return static_cast<Vec*>(vec)->data(); },
			+[](void* vec, std::size_t rows) -> void {
				static_cast<Vec*>(vec)->resize(rows, 1);
			});

	if (res == Res::failed_file) {
		terminate_with_message(CNPY_LITERAL("libnpy: could not load file"));
	}
	if (res == Res::failed_dtype) {
		terminate_with_message(CNPY_LITERAL("libnpy: mismatching scalar type"));
	}
	if (res == Res::failed_ndim) {
		char buf[4096];
		std::size_t len = std::size_t(std::snprintf(
				&buf[0],
				4096,
				"libnpy: wrong number of dimensions. expected %zu, got %zu",
				std::size_t{2},
				std::size_t(res) - std::size_t(Res::failed_ndim)));
		CNPY_ASSERT(len + 1 < 4096);
		terminate_with_message(&buf[0], len);
	}
	return out;
}

template <typename T>
auto npy_load_mat(std::string const& fname)
		-> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
	using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	Mat out;

	using Res = detail::LoadMatResult;
	using detail::terminate_with_message;

	Res res = detail::npy_vload_mat( //
			fname.c_str(),
			sizeof(T),
			std::addressof(out),
			+[](void* mat) -> void* { return static_cast<Mat*>(mat)->data(); },
			+[](void* mat, std::size_t rows, std::size_t cols) -> void {
				static_cast<Mat*>(mat)->resize(rows, cols);
			});

	if (res == Res::failed_file) {
		terminate_with_message(CNPY_LITERAL("libnpy: could not load file"));
	}
	if (res == Res::failed_dtype) {
		terminate_with_message(CNPY_LITERAL("libnpy: mismatching scalar type"));
	}
	if (res == Res::failed_ndim) {
		char buf[4096];
		std::size_t len = std::size_t(std::snprintf(
				&buf[0],
				4096,
				"libnpy: wrong number of dimensions. expected %zu, got %zu",
				std::size_t{2},
				std::size_t(res) - std::size_t(Res::failed_ndim)));
		CNPY_ASSERT(len + 1 < 4096);
		terminate_with_message(&buf[0], len);
	}
	if (res == Res::success_transpose) {
		auto rowmajor = Eigen::Map< //
				Eigen::Matrix<          //
						T,                  //
						Eigen::Dynamic,     //
						Eigen::Dynamic,     //
						Eigen::RowMajor     //
						>                   //
				>{
				out.data(),
				out.rows(),
				out.cols(),
		};
		Mat tmp = rowmajor;
		out = tmp;
	}
	return out;
}

extern template auto npy_load_mat<float>(std::string const&)
		-> Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
extern template auto npy_load_mat<double>(std::string const&)
		-> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

extern template auto npy_load_vec<float>(std::string const&)
		-> Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>;
extern template auto npy_load_vec<double>(std::string const&)
		-> Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>;

} // namespace cnpy

#endif /* end of include guard INRIA_LDLT_CNPY_HPP_TAEAIC6RS */
