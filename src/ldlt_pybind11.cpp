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
		isize start_idx) {
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
		isize row_idx) {
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

template <typename T, Layout L>
void factorize_with_permut(//
			MatRefMut<T, colmajor> out_l,
			VecRefMut<T> out_d,
			MatRef<T, L> mat
			){

			Ldlt<T> ldl{decompose, mat};
			out_d= ldl.l();
			out_l= ldl.d().asDiagonal();

}
template <typename T, Layout L>
Eigen::Matrix<T, Eigen::Dynamic, 1> iterative_solve_with_permut_fact(//
			VecRef<T> rhs,
			VecRefMut<T> sol,
			MatRef<T, L> mat,
			T eps,
			i32 max_it
			){
			Ldlt<T> ldl{decompose, mat};
			i32 it = 0;
			sol = rhs;
			ldl.solve_in_place(sol);
			auto res = (mat * sol - rhs).eval() ; 
			while(qp::infty_norm(res)>=eps){
				it +=1;
				if (it >= max_it){
					break;
				}
				res = -res ;
				ldl.solve_in_place(res);
				sol += res ;
				res = (mat * sol - rhs) ; 
			}
			return sol;
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
T initial_guess_line_search( //
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
		VecRef<T> d
		) {
		return line_search::initial_guess_line_search(
			{from_eigen, x},
			{from_eigen, ye},
			{from_eigen, ze},
			{from_eigen, dw},	
			mu_eq,
       		mu_in,
        	rho,
			QpView<T>{
				{from_eigen, H},
				{from_eigen, g},
				{from_eigen, A},
				{from_eigen, b},
				{from_eigen, C},
				{from_eigen, d},
			}
		);
}

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
		VecRef<T> l
		) {
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
			}
		);
}

template <typename T, Layout L>
T correction_guess_line_search( //
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
		VecRef<T> d
		) {
		return line_search::correction_guess_line_search(
			{from_eigen, x},
			{from_eigen, xe},
			{from_eigen, ye},
			{from_eigen, ze},
			{from_eigen, dx},	
        	mu_eq,
       		mu_in,
        	rho,
			QpView<T>{
				{from_eigen, H},
				{from_eigen, g},
				{from_eigen, A},
				{from_eigen, b},
				{from_eigen, C},
				{from_eigen, d},
			}
		);
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
		VecRef<T> l
		) {
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
			}
		);
}

Eigen::Matrix<i32, Eigen::Dynamic, 1> activeSetChange( //
		//VecRef<bool> new_active_set, does not work 
		Eigen::Matrix<bool, Eigen::Dynamic, 1>& new_active_set,
		//VecRefMut<i32> current_bijection_map,
		Eigen::Matrix<i32, Eigen::Dynamic, 1>& current_bijection_map,
		i32& n_c,
		i32& n,
        i32& n_eq,
        i32& n_in
        ) {
		return line_search::activeSetChange(
        		//ldlt::detail::nb::from_eigen_vector(new_active_set), // does not work
				new_active_set,
				current_bijection_map,
				//detail::from_eigen_vector_mut(current_bijection_map), // does not work
				n_c,
				n,
				n_eq,
				n_in
		);
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

	m.def("factorize", &ldlt::pybind11::factorize<f32, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, r>);
	m.def("factorize", &ldlt::pybind11::factorize<f32, c>);
	m.def("factorize", &ldlt::pybind11::factorize<f64, c>);

	m.def("factorize_with_permut",&ldlt::pybind11::factorize_with_permut<f32,c>);
	m.def("factorize_with_permut",&ldlt::pybind11::factorize_with_permut<f64,c>);
	m.def("iterative_solve_with_permut_fact",&ldlt::pybind11::iterative_solve_with_permut_fact<f32,c>);
	m.def("iterative_solve_with_permut_fact",&ldlt::pybind11::iterative_solve_with_permut_fact<f64,c>);

	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f32>);
	m.def("diagonal_update", &ldlt::pybind11::diagonal_update<f64>);

	m.def("row_delete", &ldlt::pybind11::row_delete<f32>);
	m.def("row_delete", &ldlt::pybind11::row_delete<f64>);

	m.def("row_append", &ldlt::pybind11::row_append<f32>);
	m.def("row_append", &ldlt::pybind11::row_append<f64>);

	m.def("solve", &ldlt::pybind11::solve<f32>);
	m.def("solve", &ldlt::pybind11::solve<f64>);


	m.def("initial_guess_line_search", &qp::pybind11::initial_guess_line_search<f32, c>);
	m.def("initial_guess_line_search", &qp::pybind11::initial_guess_line_search<f64, c>);

	m.def("initial_guess_line_search_box", &qp::pybind11::initial_guess_line_search_box<f32, c>);
	m.def("initial_guess_line_search_box", &qp::pybind11::initial_guess_line_search_box<f64, c>);

	m.def("correction_guess_line_search", &qp::pybind11::correction_guess_line_search<f32, c>);
	m.def("correction_guess_line_search", &qp::pybind11::correction_guess_line_search<f64, c>);

	m.def("correction_guess_line_search_box", &qp::pybind11::correction_guess_line_search_box<f32, c>);
	m.def("correction_guess_line_search_box", &qp::pybind11::correction_guess_line_search_box<f64, c>);

	m.def("activeSetChange", &qp::pybind11::activeSetChange);

	m.attr("__version__") = "dev";
	
}
