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

namespace cnpy {
namespace detail {

struct File
{
  File(File&& rhs) noexcept
    : ptr(rhs.ptr)
  {
    rhs.ptr = nullptr;
  }
  File(File const&) = delete;
  auto operator=(File&& rhs) noexcept -> File&
  {

    // put rhs in temporary to handle aliasing
    File tmp = CNPY_FWD(rhs);

    // clear current File
    {
      File tmp2 = static_cast<File&&>(*this);
    }

    ptr = tmp.ptr;
    tmp.ptr = nullptr;

    return *this;
  }
  auto operator=(File const&) -> File& = delete;

  std::FILE* ptr;
  File(char const* fname, char const* mode)
    : ptr{ std::fopen(fname, mode) }
  {
  }
  ~File() noexcept
  {
    if (ptr != nullptr) {
      std::fclose(ptr);
    }
  }
};

void
terminate_with_message(char const* msg, usize len)
{
  std::fwrite(msg, 1, len, stderr);
  std::fputc('\n', stderr);
  std::terminate();
}

void
append_bytes(std::vector<char>& lhs, StrView rhs)
{
  usize old_size = lhs.size();
  lhs.resize(old_size + rhs.size);
  std::memcpy(lhs.data() + old_size, rhs.data, rhs.size);
}
void
npy_vsave(char const* fname,
          void const* vdata,
          usize sizeof_T,
          char type_code,
          usize const* shape,
          usize ndim,
          char const* mode)
{

  std::FILE* fp{};
  std::vector<usize>
    true_data_shape; // if appending, the shape of existing + new data

  if (std::strcmp(mode, "a") == 0) {
    fp = std::fopen(fname, "r+b"); // NOLINT
  }

  if (fp != nullptr) {
    // file exists. we need to append to it. read the header, modify the array
    // size
    usize word_size = 0;
    bool fortran_order = false;
    detail::parse_npy_header(fp, word_size, true_data_shape, fortran_order);
    CNPY_ASSERT(!fortran_order);

    if (word_size != sizeof_T) {
      std::fprintf( // NOLINT
        stderr,
        "libnpy error: %s has word size %zu but npy_save is appending data "
        "sized %zu\n",
        fname,
        word_size,
        sizeof_T);

      CNPY_ASSERT(word_size == sizeof_T);
    }
    if (true_data_shape.size() != ndim) {
      std::fprintf( // NOLINT
        stderr,
        "libnpy error: npy_save attempting to append misdimensioned data to "
        "%s\n",
        fname);

      CNPY_ASSERT(true_data_shape.size() != ndim);
    }

    for (usize i = 1; i < ndim; i++) {
      if (shape[i] != true_data_shape[i]) {
        std::fprintf( // NOLINT
          stderr,
          "libnpy error: npy_save attempting to append misshaped data to "
          "%s\n",
          fname);

        CNPY_ASSERT(shape[i] == true_data_shape[i]);
      }
    }
    true_data_shape[0] += shape[0];
  } else {
    fp = std::fopen(fname, "wb"); // NOLINT
    true_data_shape = {};
    true_data_shape.insert(true_data_shape.end(), shape, shape + ndim);
  }

  std::vector<char> header =
    detail::create_npy_header(true_data_shape, sizeof_T, type_code);
  usize nels =
    usize(std::accumulate(shape, shape + ndim, 1, std::multiplies<usize>()));

  std::fseek(fp, 0, SEEK_SET);
  std::fwrite(&header[0], sizeof(char), header.size(), fp);
  std::fseek(fp, 0, SEEK_END);
  std::fwrite(vdata, sizeof_T, nels, fp);
  std::fclose(fp);
}
auto
create_npy_header(std::vector<usize> const& shape,
                  usize sizeof_T,
                  char type_code) -> std::vector<char>
{

  std::vector<char> dict;
  detail::append_bytes(dict, { FromLiteral{}, "{'descr': '" });
  detail::append_bytes(dict, { FromByteRepr{}, detail::BigEndianTest() });
  detail::append_bytes(dict, { FromByteRepr{}, type_code });
  {
    auto str = std::to_string(sizeof_T);
    detail::append_bytes(dict, { FromRange{}, std::to_string(sizeof_T) });
  }
  detail::append_bytes(
    dict, { FromLiteral{}, "', 'fortran_order': False, 'shape': (" });
  detail::append_bytes(dict, { FromRange{}, std::to_string(shape[0]) });
  for (usize i = 1; i < shape.size(); i++) {
    detail::append_bytes(dict, { FromLiteral{}, ", " });
    detail::append_bytes(dict, { FromRange{}, std::to_string(shape[i]) });
  }
  if (shape.size() == 1) {
    detail::append_bytes(dict, { FromLiteral{}, ", " });
  }
  detail::append_bytes(dict, { FromLiteral{}, "), }" });

  // pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10
  // bytes. dict needs to end with \n
  usize remainder = 16 - (10 + dict.size()) % 16;
  dict.insert(dict.end(), remainder, ' ');
  dict.back() = '\n';

  std::vector<char> header;
  detail::append_bytes(header, { FromLiteral{}, "\x93NUMPY\x01\x00" });
  detail::append_bytes(header, { FromByteRepr{}, uint16_t(dict.size()) });
  detail::append_bytes(header, { FromRange{}, dict });

  return header;
}

void
parse_npy_header(std::FILE* fp,
                 usize& word_size,
                 std::vector<usize>& shape,
                 bool& fortran_order)
{
  char buffer[256];
  CNPY_ASSERT(std::fread(&buffer[0], sizeof(char), 11, fp) == 11);
  std::string header = std::fgets(&buffer[0], 256, fp);
  CNPY_ASSERT(header[header.size() - 1] == '\n');

  usize loc1 = 0;
  usize loc2 = 0;

  // fortran order
  loc1 = header.find("fortran_order");
  CNPY_ASSERT(
    !(loc1 == std::string::npos) &&
    "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True");

  // shape
  loc1 = header.find('(');
  loc2 = header.find(')');
  CNPY_ASSERT(!(loc1 == std::string::npos || loc2 == std::string::npos) &&
              "parse_npy_header: failed to find header keyword: '(' or ')'");

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
  CNPY_ASSERT(!(loc1 == std::string::npos) &&
              "parse_npy_header: failed to find header keyword: 'descr'");
  loc1 += 9;
  bool littleEndian = (header[loc1] == '<' || header[loc1] == '|');
  CNPY_ASSERT(littleEndian);

  char type = header[loc1 + 1];
  CNPY_ASSERT(type == 'f');

  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find('\'');
  word_size = std::stoul(str_ws.substr(0, loc2));
}

auto
load_npy_vec(std::FILE* fp,
             usize sizeof_T,
             void* vec,
             void* (*ptr)(void*),
             void (*resize)(void*, usize rows)) -> LoadVecResult
{
  std::vector<usize> shape;

  usize word_size{};
  bool fortran_order{};

  cnpy::detail::parse_npy_header(fp, word_size, shape, fortran_order);
  if (word_size != sizeof_T) {
    return LoadVecResult::failed_dtype;
  }
  if (shape.size() != 1) {
    return LoadVecResult(int(LoadVecResult::failed_ndim) + shape.size());
  }

  usize nbytes = word_size * shape[0];
  resize(vec, shape[0]);

  CNPY_ASSERT(std::fread(ptr(vec), 1, nbytes, fp) == nbytes &&
              "load_the_npy_file: failed fread");
  return LoadVecResult::success;
}

auto
load_npy_mat(std::FILE* fp,
             usize sizeof_T,
             void* vec,
             void* (*ptr)(void*),
             void (*resize)(void*, usize rows, usize cols)) -> LoadMatResult
{
  std::vector<usize> shape;

  usize word_size{};
  bool fortran_order{};

  cnpy::detail::parse_npy_header(fp, word_size, shape, fortran_order);
  if (word_size != sizeof_T) {
    return LoadMatResult::failed_dtype;
  }
  if (shape.size() != 2) {
    return LoadMatResult(int(LoadMatResult::failed_ndim) + shape.size());
  }
  usize nbytes = word_size * shape[0] * shape[1];
  resize(vec, shape[0], shape[1]);

  CNPY_ASSERT(std::fread(ptr(vec), 1, nbytes, fp) == nbytes &&
              "load_the_npy_file: failed fread");

  if (fortran_order) {
    return LoadMatResult::success;
  } else {
    return LoadMatResult::success_transpose;
  }
}

auto
npy_vload_vec(std::string const& fname,
              usize sizeof_T,
              void* vec,
              void* (*ptr)(void*),
              void (*resize)(void*, usize rows)) -> LoadVecResult
{
  File fp{ fname.c_str(), "rb" }; // NOLINT

  CNPY_ASSERT(fp.ptr != nullptr);
  return detail::load_npy_vec(fp.ptr, sizeof_T, vec, ptr, resize);
}

auto
npy_vload_mat(std::string const& fname,
              usize sizeof_T,
              void* vec,
              void* (*ptr)(void*),
              void (*resize)(void*, usize rows, usize cols)) -> LoadMatResult
{
  File fp{ fname.c_str(), "rb" }; // NOLINT

  CNPY_ASSERT(fp.ptr != nullptr);
  return detail::load_npy_mat(fp.ptr, sizeof_T, vec, ptr, resize);
}
} // namespace detail

template auto
npy_load_mat<float>(std::string const&)
  -> Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template auto
npy_load_mat<double>(std::string const&)
  -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template auto
npy_load_vec<float>(std::string const&)
  -> Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>;
template auto
npy_load_vec<double>(std::string const&)
  -> Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>;
} // namespace cnpy
