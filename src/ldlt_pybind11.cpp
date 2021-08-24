#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace ldlt {
namespace pybind11 {

constexpr auto to_eigen_layout(Layout l) -> int {
	return l == colmajor ? Eigen::ColMajor : Eigen::RowMajor;
}
constexpr auto from_eigen_layout(int l) -> Layout {
	return (unsigned(l) & Eigen::RowMajorBit) == Eigen::RowMajor ? rowmajor
	                                                             : colmajor;
}
static_assert(
		to_eigen_layout(from_eigen_layout(Eigen::ColMajor)) == Eigen::ColMajor,
		".");
static_assert(
		to_eigen_layout(from_eigen_layout(Eigen::RowMajor)) == Eigen::RowMajor,
		".");

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

template <typename T, Layout OutL, Layout InL>
void factorize(
		MatRefMut<T, OutL> out_l, VecRefMut<T> out_d, MatRef<T, InL> mat) {
	ldlt::factorize( //
			LdltViewMut<T, OutL>{
					detail::from_eigen_matrix_mut(out_l),
					detail::from_eigen_vector_mut(out_d),
			},
			detail::from_eigen_matrix(mat),
			factorization_strategy::defer_to_colmajor);
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

     factorize_f32
     factorize_f64
  )pbdoc";
	using namespace ldlt;
	m.def("factorize", &ldlt::pybind11::factorize<f32, colmajor, colmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, rowmajor, colmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, colmajor, rowmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, rowmajor, rowmajor>);

	m.def("factorize", &ldlt::pybind11::factorize<f64, colmajor, colmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, rowmajor, colmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, colmajor, rowmajor>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, rowmajor, rowmajor>);
	m.attr("__version__") = "dev";
}
