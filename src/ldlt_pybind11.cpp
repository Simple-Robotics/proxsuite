#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace ldlt {
namespace pybind11 {

template <typename T, Layout L>
using MatRef = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)> const>;
template <typename T, Layout L>
using MatRefMut = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)>>;

template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using VecRefMut = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T, Layout L>
void factorize( //
		MatRefMut<T, colmajor> out_l,
		VecRefMut<T> out_d,
		MatRef<T, L> mat) {
	ldlt::factorize( //
			LdltViewMut<T>{
					{from_eigen, out_l},
					{from_eigen, out_d},
			},
			MatrixView<T, L>{from_eigen, mat});
}

template <typename T>
void diagonal_update( //
		MatRefMut<T, colmajor> out_l,
		VecRefMut<T> out_d,
		MatRef<T, colmajor> in_l,
		VecRef<T> in_d,
		VecRef<T> diag_section,
		i32 start_idx) {
	ldlt::diagonal_update(
			LdltViewMut<T>{
					{from_eigen, out_l},
					{from_eigen, out_d},
			},
			LdltView<T>{
					{from_eigen, in_l},
					{from_eigen, in_d},
			},
			{from_eigen, diag_section},
			start_idx);
}

template <typename T>
void row_delete( //
		MatRefMut<T, colmajor> out_l,
		VecRefMut<T> out_d,
		MatRef<T, colmajor> in_l,
		VecRef<T> in_d,
		i32 row_idx) {
	ldlt::row_delete(
			LdltViewMut<T>{
					{from_eigen, out_l},
					{from_eigen, out_d},
			},
			LdltView<T>{
					{from_eigen, in_l},
					{from_eigen, in_d},
			},
			row_idx);
}

template <typename T>
void row_append( //
		MatRefMut<T, colmajor> out_l,
		VecRefMut<T> out_d,
		MatRef<T, colmajor> in_l,
		VecRef<T> in_d,
		VecRef<T> new_row) {
	ldlt::row_append(
			LdltViewMut<T>{
					{from_eigen, out_l},
					{from_eigen, out_d},
			},
			LdltView<T>{
					{from_eigen, in_l},
					{from_eigen, in_d},
			},
			{from_eigen, new_row});
}

template <typename T>
void solve( //
		VecRefMut<T> x,
		MatRef<T, colmajor> l,
		VecRef<T> d,
		VecRef<T> rhs) {
	detail::solve_impl(
			{from_eigen, x},
			LdltView<T>{
					{from_eigen, l},
					{from_eigen, d},
			},
			{from_eigen, rhs});
}
} // namespace pybind11
} // namespace ldlt

PYBIND11_MODULE(inria_ldlt_py, m) {
	m.doc() = R"pbdoc(
INRIA LDLT decomposition
------------------------

  .. currentmodule:: inria_ldlt
  .. autosummary::
     :toctree: _generate

     factorize
  )pbdoc";
	using namespace ldlt;
	constexpr auto r = rowmajor;
	constexpr auto c = colmajor;

	m.def("factorize", &ldlt::pybind11::factorize<f32, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, c>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, c>);

	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f32>);
	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f64>);

	m.def("row_delete", &ldlt::pybind11::row_delete<f32>);
	m.def("row_delete", &ldlt::pybind11::row_delete<f64>);

	m.def("row_append", &ldlt::pybind11::row_append<f32>);
	m.def("row_append", &ldlt::pybind11::row_append<f64>);

	m.def("solve", &ldlt::pybind11::solve<f32>);
	m.def("solve", &ldlt::pybind11::solve<f64>);

	m.attr("__version__") = "dev";
}
