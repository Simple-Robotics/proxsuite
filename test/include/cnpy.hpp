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

#define CNPY_FWD(x) static_cast<decltype(x)&&>(x)

namespace cnpy {
namespace detail {
template <typename T>
struct MapType {
	static constexpr char value = '?';
};
template <>
struct MapType<float> {
	static constexpr char value = 'f';
};
template <>
struct MapType<double> {
	static constexpr char value = 'f';
};
template <>
struct MapType<long double> {
	static constexpr char value = 'f';
};
template <>
struct MapType<int> {
	static constexpr char value = 'i';
};
template <>
struct MapType<char> {
	static constexpr char value = 'i';
};
template <>
struct MapType<short> {
	static constexpr char value = 'i';
};
template <>
struct MapType<long> {
	static constexpr char value = 'i';
};
template <>
struct MapType<long long> {
	static constexpr char value = 'i';
};
template <>
struct MapType<unsigned char> {
	static constexpr char value = 'u';
};
template <>
struct MapType<unsigned short> {
	static constexpr char value = 'u';
};
template <>
struct MapType<unsigned long> {
	static constexpr char value = 'u';
};
template <>
struct MapType<unsigned long long> {
	static constexpr char value = 'u';
};
template <>
struct MapType<unsigned int> {
	static constexpr char value = 'u';
};
template <>
struct MapType<bool> {
	static constexpr char value = 'b';
};
template <>
struct MapType<std::complex<float>> {
	static constexpr char value = 'c';
};
template <>
struct MapType<std::complex<double>> {
	static constexpr char value = 'c';
};
template <>
struct MapType<std::complex<long double>> {
	static constexpr char value = 'c';
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
			: data{reinterpret_cast<char const*>(val)}, size{sizeof(T)} {}

	template <size_t N>
	constexpr StrView(FromLiteral /*tag*/, char const (&literal)[N]) noexcept
			: data{literal}, size{N - 1} {}
};
} // namespace detail

struct NpyArray {
	NpyArray(std::vector<size_t> _shape, size_t _word_size, bool _fortran_order)
			: word_size(_word_size),
				num_vals{1},
				fortran_order(_fortran_order),
				shape(CNPY_FWD(_shape)) {
		for (unsigned long i : shape) {
			num_vals *= i;
		}
		data_holder = std::vector<char>(num_vals * word_size);
	}

	NpyArray() = default;

	template <typename T>
	auto data() -> T* {
		return reinterpret_cast<T*>(data_holder.data());
	}

	template <typename T>
	auto data() const -> const T* {
		return reinterpret_cast<T*>(data_holder.data());
	}

	template <typename T>
	auto as_vec() const -> std::vector<T> {
		T const* p = data<T>();
		return std::vector<T>(p, p + num_vals);
	}

	auto num_bytes() const -> size_t { return data_holder.size(); }

	size_t word_size{};
	size_t num_vals{};
	bool fortran_order{false};
	std::vector<char> data_holder;
	std::vector<size_t> shape;
};

namespace detail {
auto BigEndianTest() -> char {
	int x = 1;
	char buf[sizeof(x)];
	std::memcpy(&buf, &x, sizeof(x));
	return buf[0] != 0 ? '<' : '>';
}
template <typename T>
auto create_npy_header(std::vector<size_t> const& shape) -> std::vector<char>;

auto create_npy_header(
		std::vector<size_t> const& shape, size_t sizeof_T, char map_value)
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
		std::string const& fname,
		void const* vdata,
		size_t sizeof_T,
		char map_value,
		std::vector<size_t> const& shape,
		std::string const& mode);
} // namespace detail
auto npy_load(std::string const& fname) -> NpyArray;

template <typename T>
void npy_save(
		std::string const& fname,
		T const* data,
		std::vector<size_t> const& shape,
		std::string const& mode = "w") {
	detail::npy_vsave(
			fname, data, sizeof(T), detail::MapType<T>::value, shape, mode);
}

template <typename T>
void npy_save(
		std::string fname, std::vector<T> const& data, std::string mode = "w") {
	std::vector<size_t> shape;
	shape.push_back(data.size());
	npy_save(fname, data.data(), shape, mode);
}

} // namespace cnpy

#endif /* end of include guard INRIA_LDLT_CNPY_HPP_TAEAIC6RS */
