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
void factorize( //
		MatRefMut<T, OutL> out_l,
		VecRefMut<T> out_d,
		MatRef<T, InL> mat) {
	ldlt::factorize( //
			LdltViewMut<T, OutL>{
					detail::from_eigen_matrix_mut(out_l),
					detail::from_eigen_vector_mut(out_d),
			},
			detail::from_eigen_matrix(mat),
			factorization_strategy::defer_to_colmajor);
}

template <typename T, Layout L>
void diagonal_update( //
		MatRefMut<T, L> out_l,
		VecRefMut<T> out_d,
		MatRef<T, L> in_l,
		VecRef<T> in_d,
		VecRef<T> diag_section,
		i32 start_idx) {
	ldlt::diagonal_update(
			LdltViewMut<T, L>{
					detail::from_eigen_matrix_mut(out_l),
					detail::from_eigen_vector_mut(out_d),
			},
			LdltView<T, L>{
					detail::from_eigen_matrix(in_l),
					detail::from_eigen_vector(in_d),
			},
			detail::from_eigen_vector(diag_section),
			start_idx);
}

template <typename T, Layout L>
void row_delete( //
		MatRefMut<T, L> out_l,
		VecRefMut<T> out_d,
		MatRef<T, L> in_l,
		VecRef<T> in_d,
		i32 row_idx) {
	ldlt::row_delete(
			LdltViewMut<T, L>{
					detail::from_eigen_matrix_mut(out_l),
					detail::from_eigen_vector_mut(out_d),
			},
			LdltView<T, L>{
					detail::from_eigen_matrix(in_l),
					detail::from_eigen_vector(in_d),
			},
			row_idx);
}

template <typename T, Layout L>
void row_append( //
		MatRefMut<T, L> out_l,
		VecRefMut<T> out_d,
		MatRef<T, L> in_l,
		VecRef<T> in_d,
		VecRef<T> new_row) {
	ldlt::row_append(
			LdltViewMut<T, L>{
					detail::from_eigen_matrix_mut(out_l),
					detail::from_eigen_vector_mut(out_d),
			},
			LdltView<T, L>{
					detail::from_eigen_matrix(in_l),
					detail::from_eigen_vector(in_d),
			},
			detail::from_eigen_vector(new_row));
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

	m.def("factorize", &ldlt::pybind11::factorize<f32, c, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, c, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, c, c>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, c, c>);

	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f32, c>);
	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f64, c>);

	m.def("row_delete", &ldlt::pybind11::row_delete<f32, c>);
	m.def("row_delete", &ldlt::pybind11::row_delete<f64, c>);

	m.def("row_append", &ldlt::pybind11::row_append<f32, c>);
	m.def("row_append", &ldlt::pybind11::row_append<f64, c>);

	m.attr("__version__") = "dev";
}
