// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at
// http://www.opensource.org/licenses/mit-license.php

#include "cnpy.hpp"
#include <sstream>
#include <complex>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cstdint>
#include <stdexcept>
#include <regex>
#include <iostream>
#include <numeric>
#include <cassert>

#define CNPY_ASSERT(Cond)                                                      \
	((Cond) ? (void)0                                                            \
	        : ::cnpy::detail::terminate_with_message(#Cond, sizeof(#Cond)))

namespace cnpy {
namespace detail {
void terminate_with_message(char const* msg, size_t len) {
	std::fwrite(msg, 1, len, stderr);
}

void append_bytes(std::vector<char>& lhs, StrView rhs) {
	size_t old_size = lhs.size();
	lhs.resize(old_size + rhs.size);
	std::memcpy(lhs.data() + old_size, rhs.data, rhs.size);
}
void npy_vsave(
		std::string const& fname,
		void const* vdata,
		size_t sizeof_T,
		char map_value,
		std::vector<size_t> const& shape,
		std::string const& mode) {

	FILE* fp = nullptr;
	std::vector<size_t>
			true_data_shape; // if appending, the shape of existing + new data

	if (mode == "a") {
		fp = fopen(fname.c_str(), "r+b"); // NOLINT
	}

	if (fp != nullptr) {
		// file exists. we need to append to it. read the header, modify the array
		// size
		size_t word_size = 0;
		bool fortran_order = false;
		detail::parse_npy_header(fp, word_size, true_data_shape, fortran_order);
		CNPY_ASSERT(!fortran_order);

		if (word_size != sizeof_T) {
			std::cout << "libnpy error: " << fname << " has word size " << word_size
								<< " but npy_save appending data sized " << sizeof_T << "\n";
			CNPY_ASSERT(word_size == sizeof_T);
		}
		if (true_data_shape.size() != shape.size()) {
			std::cout << "libnpy error: npy_save attempting to append misdimensioned "
									 "data to "
								<< fname << "\n";
			CNPY_ASSERT(true_data_shape.size() != shape.size());
		}

		for (size_t i = 1; i < shape.size(); i++) {
			if (shape[i] != true_data_shape[i]) {
				std::cout
						<< "libnpy error: npy_save attempting to append misshaped data to "
						<< fname << "\n";
				CNPY_ASSERT(shape[i] == true_data_shape[i]);
			}
		}
		true_data_shape[0] += shape[0];
	} else {
		fp = fopen(fname.c_str(), "wb"); // NOLINT
		true_data_shape = shape;
	}

	std::vector<char> header =
			detail::create_npy_header(true_data_shape, sizeof_T, map_value);
	size_t nels = size_t(std::accumulate(
			shape.begin(), shape.end(), 1, std::multiplies<size_t>()));

	fseek(fp, 0, SEEK_SET);
	fwrite(&header[0], sizeof(char), header.size(), fp);
	fseek(fp, 0, SEEK_END);
	fwrite(vdata, sizeof_T, nels, fp);
	fclose(fp);
}
auto create_npy_header(
		std::vector<size_t> const& shape, size_t sizeof_T, char map_value)
		-> std::vector<char> {

	std::vector<char> dict;
	detail::append_bytes(dict, {FromLiteral{}, "{'descr': '"});
	detail::append_bytes(dict, {FromByteRepr{}, detail::BigEndianTest()});
	detail::append_bytes(dict, {FromByteRepr{}, map_value});
	{
		auto str = std::to_string(sizeof_T);
		detail::append_bytes(dict, {FromRange{}, std::to_string(sizeof_T)});
	}
	detail::append_bytes(
			dict, {FromLiteral{}, "', 'fortran_order': False, 'shape': ("});
	detail::append_bytes(dict, {FromRange{}, std::to_string(shape[0])});
	for (size_t i = 1; i < shape.size(); i++) {
		detail::append_bytes(dict, {FromLiteral{}, ", "});
		detail::append_bytes(dict, {FromRange{}, std::to_string(shape[i])});
	}
	if (shape.size() == 1) {
		detail::append_bytes(dict, {FromLiteral{}, ", "});
	}
	detail::append_bytes(dict, {FromLiteral{}, "), }"});

	// pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10
	// bytes. dict needs to end with \n
	size_t remainder = 16 - (10 + dict.size()) % 16;
	dict.insert(dict.end(), remainder, ' ');
	dict.back() = '\n';

	std::vector<char> header;
	detail::append_bytes(header, {FromByteRepr{}, char(0x93)});
	detail::append_bytes(header, {FromLiteral{}, "NUMPY"});
	detail::append_bytes(header, {FromByteRepr{}, char(0x01)});
	detail::append_bytes(header, {FromByteRepr{}, char(0x00)});
	detail::append_bytes(header, {FromByteRepr{}, uint16_t(dict.size())});
	detail::append_bytes(header, {FromRange{}, dict});

	return header;
}

void parse_npy_header(
		FILE* fp,
		size_t& word_size,
		std::vector<size_t>& shape,
		bool& fortran_order) {
	char buffer[256];
	size_t res = fread(&buffer[0], sizeof(char), 11, fp);
	if (res != 11) {
		throw std::runtime_error("parse_npy_header: failed fread");
	}
	std::string header = fgets(&buffer[0], 256, fp);
	CNPY_ASSERT(header[header.size() - 1] == '\n');

	size_t loc1 = 0;
	size_t loc2 = 0;

	// fortran order
	loc1 = header.find("fortran_order");
	if (loc1 == std::string::npos) {
		throw std::runtime_error(
				"parse_npy_header: failed to find header keyword: 'fortran_order'");
	}
	loc1 += 16;
	fortran_order = (header.substr(loc1, 4) == "True");

	// shape
	loc1 = header.find('(');
	loc2 = header.find(')');
	if (loc1 == std::string::npos || loc2 == std::string::npos) {
		throw std::runtime_error(
				"parse_npy_header: failed to find header keyword: '(' or ')'");
	}

	std::regex num_regex("[0-9][0-9]*");
	std::smatch sm;
	shape.clear();

	std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
	while (std::regex_search(str_shape, sm, num_regex)) {
		shape.push_back(std::stoul(sm[0].str()));
		str_shape = sm.suffix().str();
	}

	// endian, word size, data type
	// byte order code | stands for not applicable.
	// not sure when this applies except for byte array
	loc1 = header.find("descr");
	if (loc1 == std::string::npos) {
		throw std::runtime_error(
				"parse_npy_header: failed to find header keyword: 'descr'");
	}
	loc1 += 9;
	bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
	CNPY_ASSERT(littleEndian);

	// char type = header[loc1+1];
	// CNPY_ASSERT(type == map_type(T));

	std::string str_ws = header.substr(loc1 + 2);
	loc2 = str_ws.find('\'');
	word_size = std::stoul(str_ws.substr(0, loc2));
}
auto load_the_npy_file(FILE* fp) -> cnpy::NpyArray {
	std::vector<size_t> shape;
	size_t word_size = 0;
	bool fortran_order = false;
	cnpy::detail::parse_npy_header(fp, word_size, shape, fortran_order);

	cnpy::NpyArray arr(shape, word_size, fortran_order);
	size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
	if (nread != arr.num_bytes()) {
		throw std::runtime_error("load_the_npy_file: failed fread");
	}
	return arr;
}
} // namespace detail
} // namespace cnpy

auto cnpy::npy_load(std::string const& fname) -> cnpy::NpyArray {

	FILE* fp = fopen(fname.c_str(), "rb"); // NOLINT

	if (fp == nullptr) {
		throw std::runtime_error("npy_load: Unable to open file " + fname);
	}
	NpyArray arr = detail::load_the_npy_file(fp);

	fclose(fp);
	return arr;
}
