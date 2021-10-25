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
    res = -rhs;
    res.noalias() += mat * sol;
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
auto initial_guess_line_search_box( //
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
		VecRef<T> l) -> T {
	return line_search::initial_guess_line_search_box(
			{from_eigen, x.eval()},
			{from_eigen, ye.eval()},
			{from_eigen, ze.eval()},
			{from_eigen, dw.eval()},
			mu_eq,
			mu_in,
			rho,
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			});
}

template <typename T, Layout L>
auto correction_guess_line_search_box( //
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
		VecRef<T> l) -> T {
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

void active_set_change(
		VecRef<bool> const& new_active_set,
		VecRefMut<isize> current_bijection_map,
		isize n_c,
		isize n_in) {
	return line_search::active_set_change(
			{from_eigen, new_active_set},
			{from_eigen, current_bijection_map},
			n_c,
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
