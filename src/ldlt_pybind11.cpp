#include <ldlt/ldlt.hpp>
#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>
#include <ldlt/update.hpp>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <qp/line_search.hpp>
#include <qp/views.hpp>
#include <qp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>

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
void QPsolve( //
		VecRefMut<T> x,
		VecRefMut<T> y,
		VecRefMut<T> z,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		isize max_iter,
		isize max_iter_in,
		VecRefMut<double> res_iter,
		T eps_abs,
		T eps_rel,
		T eps_IG,
		T beta,
		T R) {

	isize dim = H.eval().rows();
	isize n_eq = A.eval().rows();
	isize n_in = C.eval().rows();
	auto ruiz = qp::preconditioner::RuizEquilibration<T>{
			dim,
			n_eq + n_in,
	};
	qp::detail::QpSolveStats res = qp::detail::qpSolve( //
			{from_eigen, x},
			{from_eigen, y},
			{from_eigen, z},
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			},
			max_iter,
			max_iter_in,
			eps_abs,
			eps_rel,
			eps_IG,
			beta,
			R,
			LDLT_FWD(ruiz));

	std::cout << "------ SOLVER STATISTICS--------" << std::endl;
	std::cout << "n_ext : " << res.n_ext << std::endl;
	std::cout << "n_tot : " << res.n_tot << std::endl;
	std::cout << "mu updates : " << res.n_mu_updates << std::endl;

	res_iter(0) = double(res.n_ext);
	res_iter(1) = double(res.n_tot);
	res_iter(2) = double(res.n_mu_updates);
	/*
	res_iter(3) = double(res.equilibration_tmp);
	res_iter(4) = double(res.fact_tmp );
	res_iter(5) = double(res.ws_tmp ); 
	res_iter(6) = double(res.residuals_tmp) ;
	res_iter(7) = double(res.IG_tmp );
	res_iter(8) = double(res.CG_tmp );
	res_iter(9) = double(res.BCL_tmp );
	res_iter(10) = double(res.cold_restart_tmp) ; 
	*/

}

template <typename T, Layout L>
void QPalmSolve( //
		VecRefMut<T> x,
		VecRefMut<T> y,
		VecRefMut<T> z,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		isize max_iter,
		isize max_iter_in,
		VecRefMut<double> res_iter,
		T eps_abs,
		T eps_rel,
		T max_rank_update,
		T max_rank_update_fraction) {

	isize dim = H.eval().rows();
	isize n_eq = A.eval().rows();
	isize n_in = C.eval().rows();
	auto ruiz = qp::preconditioner::RuizEquilibration<T>{
			dim,
			n_eq + n_in,
	};
	qp::detail::QpalmSolveStats res = qp::detail::QPALMSolve( //
			{from_eigen, x},
			{from_eigen, y},
			{from_eigen, z},
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			},
			max_iter,
			max_iter_in,
			eps_abs,
			eps_rel,
			max_rank_update,
			max_rank_update_fraction,
			LDLT_FWD(ruiz));

	std::cout << "------ SOLVER STATISTICS--------" << std::endl;
	std::cout << "n_ext : " << res.n_ext << std::endl;
	std::cout << "n_tot : " << res.n_tot << std::endl;
	std::cout << "mu updates : " << res.n_mu_updates << std::endl;

	res_iter(0) = double(res.n_ext);
	res_iter(1) = double(res.n_tot);
	res_iter(2) = double(res.n_mu_updates);
}

template <typename T, Layout L>
void OSQPsolve( //
		VecRefMut<T> x,
		VecRefMut<T> y,
		MatRef<T, L> H,
		VecRef<T> g,
		MatRef<T, L> A,
		VecRef<T> b,
		MatRef<T, L> C,
		VecRef<T> u,
		VecRef<T> l,
		isize max_iter,
		isize max_iter_in,
		VecRefMut<T> res_iter,
		T eps_abs,
		T eps_rel) {

	isize dim = H.eval().rows();
	isize n_eq = A.eval().rows();
	isize n_in = C.eval().rows();
	auto ruiz = qp::preconditioner::RuizEquilibration<T>{
			dim,
			n_eq + n_in,
	};
	qp::detail::QpSolveStats res = qp::detail::osqpSolve( //
			{from_eigen, x},
			{from_eigen, y},
			QpViewBox<T>{
					{from_eigen, H.eval()},
					{from_eigen, g.eval()},
					{from_eigen, A.eval()},
					{from_eigen, b.eval()},
					{from_eigen, C.eval()},
					{from_eigen, u.eval()},
					{from_eigen, l.eval()},
			},
			max_iter,
			max_iter_in,
			eps_abs,
			eps_rel,
			LDLT_FWD(ruiz));

	std::cout << "------ SOLVER STATISTICS--------" << std::endl;
	std::cout << "n_ext : " << res.n_ext << std::endl;
	std::cout << "n_tot : " << res.n_tot << std::endl;
	std::cout << "mu updates : " << res.n_mu_updates << std::endl;

	res_iter(0) = T(res.n_ext);
	res_iter(1) = T(res.n_tot);
	res_iter(2) = T(res.n_mu_updates);

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
	// using namespace preconditioner;
	constexpr auto c = colmajor;

	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f32, c>);
	m.def(
			"iterative_solve_with_permut_fact",
			&ldlt::pybind11::iterative_solve_with_permut_fact<f64, c>);

	m.def("QPalmSolve", &qp::pybind11::QPalmSolve<f32, c>);
	m.def("QPalmSolve", &qp::pybind11::QPalmSolve<f64, c>);

	m.def("QPsolve", &qp::pybind11::QPsolve<f32, c>);
	m.def("QPsolve", &qp::pybind11::QPsolve<f64, c>);

	m.def("OSQPsolve", &qp::pybind11::OSQPsolve<f32, c>);
	m.def("OSQPsolve", &qp::pybind11::OSQPsolve<f64, c>);

	m.attr("__version__") = "dev";
}
