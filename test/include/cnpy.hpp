// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at
// http://www.opensource.org/licenses/mit-license.php

#ifndef PROXSUITE_TEST_CNPY_HPP
#define PROXSUITE_TEST_CNPY_HPP

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
  ((Cond)                                                                      \
     ? (void)0                                                                 \
     : ::cnpy::detail::terminate_with_message(                                 \
         "assertion failed: " #Cond, sizeof("assertion failed: " #Cond) - 1))

#define CNPY_LITERAL(X) X, sizeof(X) - 1

namespace cnpy {
using usize = decltype(sizeof(0));
namespace detail {

enum struct LoadVecResult : int
{
  success = 0,
  failed_file = 1,
  failed_dtype = 2,
  failed_ndim = 3,
};

enum struct LoadMatResult : int
{
  success = 0,
  success_transpose = -1,

  failed_file = 1,
  failed_dtype = 2,
  failed_ndim = 3,
};

void
terminate_with_message(char const* msg, usize len);

template<typename T>
struct TypeCode;
template<>
struct TypeCode<float>
{
  static constexpr char value = 'f';
};
template<>
struct TypeCode<double>
{
  static constexpr char value = 'f';
};

struct FromByteRepr
{};
struct FromLiteral
{};
struct FromRawParts
{};
struct FromRange
{};

struct StrView
{
  char const* data;
  usize size;

  StrView(FromRawParts /*tag*/, char const* data_, usize size_) noexcept
    : data{ data_ }
    , size{ size_ }
  {
  }

  template<typename T>
  StrView(FromRange /*tag*/, T const& str) noexcept
    : data{ str.data() }
    , size{ str.size() }
  {
  }

  template<typename T>
  StrView(FromByteRepr /*tag*/, T const& val) noexcept
    : data{ &reinterpret_cast<char const&>(val) }
    , size{ sizeof(T) }
  {
  }

  template<usize N>
  constexpr StrView(FromLiteral /*tag*/, char const (&literal)[N]) noexcept
    : data{ literal }
    , size{ N - 1 }
  {
  }
};

} // namespace detail

namespace detail {
inline auto
BigEndianTest() -> char
{
  int x = 1;
  char buf[sizeof(x)];
  std::memcpy(&buf, &x, sizeof(x));
  return buf[0] != 0 ? '<' : '>';
}
template<typename T>
auto
create_npy_header(std::vector<usize> const& shape) -> std::vector<char>;

auto
create_npy_header(std::vector<usize> const& shape,
                  usize sizeof_T,
                  char type_code) -> std::vector<char>;
void
parse_npy_header(FILE* fp,
                 usize& word_size,
                 std::vector<usize>& shape,
                 bool& fortran_order);
void
parse_npy_header(unsigned char* buffer,
                 usize& word_size,
                 std::vector<usize>& shape,
                 bool& fortran_order);

void
npy_vsave(char const* fname,
          void const* vdata,
          usize sizeof_T,
          char type_code,
          usize const* shape,
          usize ndim,
          char const* mode);

auto
npy_vload_vec(std::string const& fname,
              usize sizeof_T,
              void* vec,
              void* (*ptr)(void*),
              void (*resize)(void*, usize rows)) -> LoadVecResult;
auto
npy_vload_mat(std::string const& fname,
              usize sizeof_T,
              void* vec,
              void* (*ptr)(void*),
              void (*resize)(void*, usize rows, usize cols)) -> LoadMatResult;
} // namespace detail

template<typename D>
void
npy_save_mat(std::string const& fname,
             Eigen::MatrixBase<D> const& mat,
             std::string const& mode = "w")
{

  auto const& eval = mat.eval();
  usize rowcol[] = { usize(eval.rows()), usize(eval.cols()) };

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

template<typename D>
void
npy_save_vec(std::string const& fname,
             Eigen::MatrixBase<D> const& vec,
             const std::string& mode = "w")
{

  auto const& eval = vec.eval();
  using T = typename D::Scalar;
  CNPY_ASSERT(vec.cols() == 1);
  usize size = vec.rows();

  detail::npy_vsave( //
    fname.c_str(),
    vec.data(),
    sizeof(T),
    detail::TypeCode<T>::value,
    &size,
    1,
    mode.c_str());
}

template<typename T>
auto
npy_load_vec(std::string const& fname) -> Eigen::Matrix<T, Eigen::Dynamic, 1>
{
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vec out;

  using Res = detail::LoadVecResult;
  using detail::terminate_with_message;

  Res res = detail::npy_vload_vec( //
    fname.c_str(),
    sizeof(T),
    std::addressof(out),
    +[](void* vec) -> void* { return static_cast<Vec*>(vec)->data(); },
    +[](void* vec, usize rows) -> void {
      static_cast<Vec*>(vec)->resize(Eigen::Index(rows), 1);
    });

  if (res == Res::failed_file) {
    terminate_with_message(CNPY_LITERAL("libnpy: could not load file"));
  }
  if (res == Res::failed_dtype) {
    terminate_with_message(CNPY_LITERAL("libnpy: mismatching scalar type"));
  }
  if (res == Res::failed_ndim) {
    char buf[4096];
    usize len = usize(
      std::snprintf(&buf[0],
                    4096,
                    "libnpy: wrong number of dimensions. expected %zu, got %zu",
                    usize{ 2 },
                    usize(res) - usize(Res::failed_ndim)));
    CNPY_ASSERT(len + 1 < 4096);
    terminate_with_message(&buf[0], len);
  }
  return out;
}

template<typename T>
auto
npy_load_mat(std::string const& fname)
  -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
{
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  Mat out;

  using Res = detail::LoadMatResult;
  using detail::terminate_with_message;

  Res res = detail::npy_vload_mat( //
    fname.c_str(),
    sizeof(T),
    std::addressof(out),
    +[](void* mat) -> void* { return static_cast<Mat*>(mat)->data(); },
    +[](void* mat, usize rows, usize cols) -> void {
      static_cast<Mat*>(mat)->resize(Eigen::Index(rows), Eigen::Index(cols));
    });

  if (res == Res::failed_file) {
    terminate_with_message(CNPY_LITERAL("libnpy: could not load file"));
  }
  if (res == Res::failed_dtype) {
    terminate_with_message(CNPY_LITERAL("libnpy: mismatching scalar type"));
  }
  if (res == Res::failed_ndim) {
    char buf[4096];
    usize len = usize(
      std::snprintf(&buf[0],
                    4096,
                    "libnpy: wrong number of dimensions. expected %zu, got %zu",
                    usize{ 2 },
                    usize(res) - usize(Res::failed_ndim)));
    CNPY_ASSERT(len + 1 < 4096);
    terminate_with_message(&buf[0], len);
  }
  if (res == Res::success_transpose) {
    auto rowmajor = Eigen::Map< //
      Eigen::Matrix<            //
        T,                      //
        Eigen::Dynamic,         //
        Eigen::Dynamic,         //
        Eigen::RowMajor         //
        >                       //
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

extern template auto
npy_load_mat<float>(std::string const&)
  -> Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
extern template auto
npy_load_mat<double>(std::string const&)
  -> Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

extern template auto
npy_load_vec<float>(std::string const&)
  -> Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::ColMajor>;
extern template auto
npy_load_vec<double>(std::string const&)
  -> Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>;

} // namespace cnpy

#endif /* end of include guard PROXSUITE_TEST_CNPY_HPP */
