#include <qp/views.hpp>
#include <qp/QPWorkspace.hpp>

#include <qp/proxqp/solver.hpp>
#include <qp/utils.hpp>
#include <qp/precond/ruiz.hpp>
#include <fmt/chrono.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <veg/util/dynstack_alloc.hpp>
#include <dense-ldlt/ldlt.hpp>

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

} // namespace pybind11
} // namespace ldlt

namespace qp {
namespace pybind11 {

using namespace ldlt::tags;
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
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T, Layout L>
void QPupdateMatrice( //
		MatRef<T, L> H,
		MatRef<T, L> A,
		MatRef<T, L> C,
		const qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,
		const bool update = false) {

	// update matrices

	qpmodel.H = H.eval();
	qpmodel.A = A.eval();
	qpmodel.C = C.eval();

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qp::QpViewBoxMut<T> qp_scaled{
			{from_eigen, qpwork.H_scaled},
			{from_eigen, qpwork.g_scaled},
			{from_eigen, qpwork.A_scaled},
			{from_eigen, qpwork.b_scaled},
			{from_eigen, qpwork.C_scaled},
			{from_eigen, qpwork.u_scaled},
			{from_eigen, qpwork.l_scaled}};

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut,
			qpwork.ldl_stack.as_mut(),
	};
	qpwork.ruiz.scale_qp_in_place(qp_scaled, stack);

	qpwork.dw_aug.setZero();

	// re update all other variables
	if (update) {
		QPResults.reset_results();
	}

	// perform warm start

	qpwork.primal_feasibility_rhs_1_eq = infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			QPResults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-QPResults.mu_eq_inv); // mu stores the inverse of mu

	qpwork.ldl.factorize(qpwork.kkt, stack);

	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	qp::detail::iterative_solve_with_permut_fact( //
			QPSettings,
			qpmodel,
			QPResults,
			qpwork,
			T(1),
			qpmodel.dim + qpmodel.n_eq);

	QPResults.x = qpwork.dw_aug.head(qpmodel.dim);
	QPResults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

	qpwork.dw_aug.setZero();
}

template <typename T>
void QPupdateVectors( //
		VecRef<T> g,
		VecRef<T> b,
		VecRef<T> u,
		VecRef<T> l,
		const qp::QPSettings<T>& QPSettings,
		qp::QPData<T>& qpmodel,
		qp::QPWorkspace<T>& qpwork,
		qp::QPResults<T>& QPResults,
		const bool update = false) {

	// update vectors

	qpmodel.g = g.eval();
	qpmodel.b = b.eval();
	qpmodel.u = u.eval();
	qpmodel.l = l.eval();

	qpwork.g_scaled = qpmodel.g;
	qpwork.b_scaled = qpmodel.b;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

	qpwork.ruiz.scale_primal_in_place(
			VectorViewMut<T>{from_eigen, qpwork.g_scaled});
	qpwork._scale_dual_in_place_eq(VectorViewMut<T>{from_eigen, qpwork.b_scaled});
	qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.u_scaled});
	qpwork._scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.l_scaled});

}

template <typename T, Layout L>
void QPsolve(
		const qp::QPData<T>& qpmodel,
		qp::QPResults<T>& QPResults,
		qp::QPWorkspace<T>& qpwork,
		const qp::QPSettings<T>& qpsettings) {

	auto start = std::chrono::high_resolution_clock::now();
	qp::detail::qp_solve( //
			qpsettings,
			qpmodel,
			QPResults,
			qpwork);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	QPResults.timing = duration.count();

	if (qpsettings.verbose) {
		std::cout << "------ SOLVER STATISTICS--------" << std::endl;
		std::cout << "n_ext : " << QPResults.n_ext << std::endl;
		std::cout << "n_tot : " << QPResults.n_tot << std::endl;
		std::cout << "mu updates : " << QPResults.n_mu_change << std::endl;
		std::cout << "objValue : " << QPResults.objValue << std::endl;
		std::cout << "timing : " << QPResults.timing << std::endl;
	}
}

template <typename T>
void QPreset(
		const qp::QPData<T>& qpmodel,
		const qp::QPSettings<T>& qpsettings,
		qp::QPResults<T>& qpresults,
		qp::QPWorkspace<T>& qpwork) {

	qpwork.kkt.diagonal().head(qpmodel.dim).array() -= qpresults.rho;
	qpresults.reset_results();
	qpwork.reset_results(qpmodel.n_in);

	qpwork.kkt.diagonal().head(qpmodel.dim).array() += qpresults.rho;
	qpwork.kkt.diagonal().segment(qpmodel.dim, qpmodel.n_eq).array() =
			-qpresults.mu_eq_inv;

	veg::dynstack::DynStackMut stack{
			veg::from_slice_mut,
			qpwork.ldl_stack.as_mut(),
	};
	qpwork.ldl.factorize(qpwork.kkt, stack);

	if (qpsettings.warm_start){
		qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
		qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

		qp::detail::iterative_solve_with_permut_fact( //
				qpsettings,
				qpmodel,
				qpresults,
				qpwork,
				T(1),
				qpmodel.dim + qpmodel.n_eq);

		qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
		qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

		qpwork.dw_aug.setZero();
		qpwork.rhs.setZero();
	}
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
	::pybind11::class_<qp::QPWorkspace<f64>>(m, "QPWorkspace")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member
			.def_readwrite("H_scaled", &qp::QPWorkspace<f64>::H_scaled)
			.def_readwrite("g_scaled", &qp::QPWorkspace<f64>::g_scaled)
			.def_readwrite("A_scaled", &qp::QPWorkspace<f64>::A_scaled)
			.def_readwrite("C_scaled", &qp::QPWorkspace<f64>::C_scaled)
			.def_readwrite("b_scaled", &qp::QPWorkspace<f64>::b_scaled)
			.def_readwrite("u_scaled", &qp::QPWorkspace<f64>::u_scaled)
			.def_readwrite("l_scaled", &qp::QPWorkspace<f64>::l_scaled)
			.def_readwrite("x_prev", &qp::QPWorkspace<f64>::x_prev)
			.def_readwrite("y_prev", &qp::QPWorkspace<f64>::y_prev)
			.def_readwrite("z_prev", &qp::QPWorkspace<f64>::z_prev)
			.def_readwrite("kkt", &qp::QPWorkspace<f64>::kkt)
			.def_readwrite(
					"current_bijection_map", &qp::QPWorkspace<f64>::current_bijection_map)
			.def_readwrite(
					"new_bijection_map", &qp::QPWorkspace<f64>::new_bijection_map)
			.def_readwrite("active_set_up", &qp::QPWorkspace<f64>::active_set_up)
			.def_readwrite("active_set_low", &qp::QPWorkspace<f64>::active_set_low)
			.def_readwrite(
					"active_inequalities", &qp::QPWorkspace<f64>::active_inequalities)
			.def_readwrite("Hdx", &qp::QPWorkspace<f64>::Hdx)
			.def_readwrite("Cdx", &qp::QPWorkspace<f64>::Cdx)
			.def_readwrite("Adx", &qp::QPWorkspace<f64>::Adx)
			.def_readwrite("active_part_z", &qp::QPWorkspace<f64>::active_part_z)
			.def_readwrite("alphas", &qp::QPWorkspace<f64>::alphas)
			.def_readwrite("dw_aug", &qp::QPWorkspace<f64>::dw_aug)
			.def_readwrite("rhs", &qp::QPWorkspace<f64>::rhs)
			.def_readwrite("err", &qp::QPWorkspace<f64>::err)
			.def_readwrite(
					"primal_feasibility_rhs_1_eq",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_eq)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_u",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_in_u)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_l",
					&qp::QPWorkspace<f64>::primal_feasibility_rhs_1_in_l)
			.def_readwrite(
					"dual_feasibility_rhs_2",
					&qp::QPWorkspace<f64>::dual_feasibility_rhs_2)
			.def_readwrite(
					"correction_guess_rhs_g",
					&qp::QPWorkspace<f64>::correction_guess_rhs_g)
			.def_readwrite("alpha", &qp::QPWorkspace<f64>::alpha)
			.def_readwrite(
					"dual_residual_scaled", &qp::QPWorkspace<f64>::dual_residual_scaled)
			.def_readwrite(
					"primal_residual_eq_scaled",
					&qp::QPWorkspace<f64>::primal_residual_eq_scaled)
			.def_readwrite(
					"primal_residual_in_scaled_up",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_up)
			.def_readwrite(
					"primal_residual_in_scaled_low",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_low)
			.def_readwrite(
					"primal_residual_in_scaled_up_plus_alphaCdx",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_up_plus_alphaCdx)
			.def_readwrite(
					"primal_residual_in_scaled_low_plus_alphaCdx",
					&qp::QPWorkspace<f64>::primal_residual_in_scaled_low_plus_alphaCdx)
			.def_readwrite("CTz", &qp::QPWorkspace<f64>::CTz);

	::pybind11::class_<qp::QPResults<f64>>(m, "QPResults")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member

			.def_readwrite("x", &qp::QPResults<f64>::x)
			.def_readwrite("y", &qp::QPResults<f64>::y)
			.def_readwrite("z", &qp::QPResults<f64>::z)
			.def_readwrite("n_c", &qp::QPResults<f64>::n_c)
			.def_readwrite("mu_eq", &qp::QPResults<f64>::mu_eq)
			.def_readwrite("mu_eq_inv", &qp::QPResults<f64>::mu_eq_inv)
			.def_readwrite("mu_in", &qp::QPResults<f64>::mu_in)
			.def_readwrite("mu_in_inv", &qp::QPResults<f64>::mu_in_inv)
			.def_readwrite("rho", &qp::QPResults<f64>::rho)
			.def_readwrite("n_tot", &qp::QPResults<f64>::n_tot)
			.def_readwrite("n_ext", &qp::QPResults<f64>::n_ext)
			.def_readwrite("timing", &qp::QPResults<f64>::timing)
			.def_readwrite("objValue", &qp::QPResults<f64>::objValue)
			.def_readwrite("n_mu_change", &qp::QPResults<f64>::n_mu_change);

	::pybind11::class_<qp::QPSettings<f64>>(m, "QPSettings")
			.def(::pybind11::init()) // constructor
	                             // read-write public data member

			.def_readwrite("alpha_bcl", &qp::QPSettings<f64>::alpha_bcl)
			.def_readwrite("beta_bcl", &qp::QPSettings<f64>::beta_bcl)
			.def_readwrite(
					"refactor_dual_feasibility_threshold",
					&qp::QPSettings<f64>::refactor_dual_feasibility_threshold)
			//.def_readwrite("pmm", &qp::QPSettings<f64>::pmm)
			.def_readwrite(
					"refactor_rho_threshold",
					&qp::QPSettings<f64>::refactor_rho_threshold)
			.def_readwrite("mu_max_eq", &qp::QPSettings<f64>::mu_max_eq)
			.def_readwrite("mu_max_in", &qp::QPSettings<f64>::mu_max_in)
			.def_readwrite("mu_max_eq_inv", &qp::QPSettings<f64>::mu_max_eq_inv)
			.def_readwrite("mu_max_in_inv", &qp::QPSettings<f64>::mu_max_in_inv)
			.def_readwrite("mu_update_factor", &qp::QPSettings<f64>::mu_update_factor)
			.def_readwrite(
					"mu_update_inv_factor", &qp::QPSettings<f64>::mu_update_inv_factor)

			.def_readwrite("cold_reset_mu_eq", &qp::QPSettings<f64>::cold_reset_mu_eq)
			.def_readwrite("cold_reset_mu_in", &qp::QPSettings<f64>::cold_reset_mu_in)
			.def_readwrite(
					"cold_reset_mu_eq_inv", &qp::QPSettings<f64>::cold_reset_mu_eq_inv)

			.def_readwrite(
					"cold_reset_mu_in_inv", &qp::QPSettings<f64>::cold_reset_mu_in_inv)
			.def_readwrite("max_iter", &qp::QPSettings<f64>::max_iter)
			.def_readwrite("max_iter_in", &qp::QPSettings<f64>::max_iter_in)

			.def_readwrite("eps_abs", &qp::QPSettings<f64>::eps_abs)
			.def_readwrite("eps_rel", &qp::QPSettings<f64>::eps_rel)
			.def_readwrite("eps_IG", &qp::QPSettings<f64>::eps_IG)
			.def_readwrite("R", &qp::QPSettings<f64>::R)
			.def_readwrite(
					"nb_iterative_refinement",
					&qp::QPSettings<f64>::nb_iterative_refinement)
			.def_readwrite("warm_start", &qp::QPSettings<f64>::warm_start)
			.def_readwrite("verbose", &qp::QPSettings<f64>::verbose);

	::pybind11::class_<qp::QPData<f64>>(m, "QPData")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member

			.def_readonly("H", &qp::QPData<f64>::H)
			.def_readonly("g", &qp::QPData<f64>::g)
			.def_readonly("A", &qp::QPData<f64>::A)
			.def_readonly("b", &qp::QPData<f64>::b)
			.def_readonly("C", &qp::QPData<f64>::C)
			.def_readonly("u", &qp::QPData<f64>::u)
			.def_readonly("l", &qp::QPData<f64>::l)
			.def_readonly("dim", &qp::QPData<f64>::dim)
			.def_readonly("n_eq", &qp::QPData<f64>::n_eq)
			.def_readonly("n_in", &qp::QPData<f64>::n_in)
			.def_readonly("n_total", &qp::QPData<f64>::n_total);

	constexpr auto c = rowmajor;

	m.def("QPsolve", &qp::pybind11::QPsolve<f32, c>);
	m.def("QPsolve", &qp::pybind11::QPsolve<f64, c>);

	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f32, c>);
	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f64, c>);

	m.def("QPsetup", &qp::detail::QPsetup<f32>);
	m.def("QPsetup", &qp::detail::QPsetup<f64>);

	m.def("QPreset", &qp::pybind11::QPreset<f32>);
	m.def("QPreset", &qp::pybind11::QPreset<f64>);

	m.attr("__version__") = "dev";
}
