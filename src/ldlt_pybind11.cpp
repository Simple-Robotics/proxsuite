#include <ldlt/ldlt.hpp>
#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/line_search.hpp>
#include <qp/views.hpp>

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
void iterative_solve_with_permut_fact( //
		VecRef<T> rhs,
		VecRefMut<T> sol,
		MatRef<T, L> mat,
		T eps,
		i32 max_it) {
	Ldlt<T> ldl{decompose, mat};
	i32 it = 0;
	sol = rhs;
	ldl.solve_in_place(sol);
	auto res = (mat * sol - rhs).eval();
	while (qp::infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;
		res = (mat * sol - rhs);
	}
}

} // namespace pybind11
} // namespace ldlt

namespace qp {
namespace pybind11 {

using namespace ldlt;
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
T initial_guess_line_search_box( //
		VecRef<T> x,
		VecRef<T> ye,
		VecRef<T> ze,
		VecRef<T> dw,
		T mu_eq,
		T mu_in,
		T rho,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l) {
	return line_search::initial_guess_line_search_box(
			{from_eigen, x},
			{from_eigen, ye},
			{from_eigen, ze},
			{from_eigen, dw},
			mu_eq,
			mu_in,
			rho,
			QpViewBox<T>{
					{from_eigen, H},
					{from_eigen, g},
					{from_eigen, A},
					{from_eigen, b},
					{from_eigen, C},
					{from_eigen, u},
					{from_eigen, l},
			});
}

template <typename T, Layout L>
T correction_guess_line_search_box( //
		VecRef<T> x,
		VecRef<T> xe,
		VecRef<T> ye,
		VecRef<T> ze,
		VecRef<T> dx,
		T mu_eq,
		T mu_in,
		T rho,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l) {
	return line_search::correction_guess_line_search_box(
			{from_eigen, x},
			{from_eigen, xe},
			{from_eigen, ye},
			{from_eigen, ze},
			{from_eigen, dx},
			mu_eq,
			mu_in,
			rho,
			QpViewBox<T>{
					{from_eigen, H},
					{from_eigen, g},
					{from_eigen, A},
					{from_eigen, b},
					{from_eigen, C},
					{from_eigen, u},
					{from_eigen, l},
			});
}

Eigen::Matrix<i32, Eigen::Dynamic, 1>
active_set_change( //
                   // VecRef<bool> new_active_set, does not work
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& new_active_set,
		// VecRefMut<i32> current_bijection_map,
		Eigen::Matrix<i32, Eigen::Dynamic, 1>& current_bijection_map,
		i32& n_c,
		i32& n,
		i32& n_eq,
		i32& n_in) {
	return line_search::active_set_change(
			// ldlt::detail::nb::from_eigen_vector(new_active_set), // does not work
			new_active_set,
			current_bijection_map,
			// detail::from_eigen_vector_mut(current_bijection_map), // does not work
			n_c,
			n,
			n_eq,
			n_in);
}

} // namespace pybind11
} // namespace qp

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
	using namespace qp;
	constexpr auto r = rowmajor;
	constexpr auto c = colmajor;

	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f32, c>);
	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f64, c>);

	m.def(
			"initial_guess_line_search_box",
			&qp::pybind11::initial_guess_line_search_box<f32, c>);
	m.def(
			"initial_guess_line_search_box",
			&qp::pybind11::initial_guess_line_search_box<f64, c>);

	m.def(
			"correction_guess_line_search_box",
			&qp::pybind11::correction_guess_line_search_box<f32, c>);
	m.def(
			"correction_guess_line_search_box",
			&qp::pybind11::correction_guess_line_search_box<f64, c>);

	m.def("active_set_change", &qp::pybind11::active_set_change);

	m.attr("__version__") = "dev";
}
