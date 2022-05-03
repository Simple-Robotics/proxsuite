#include <qp/dense/dense-views.hpp>
#include <qp/dense/Workspace.hpp>

#include <qp/dense/solver.hpp>
#include <qp/dense/dense-utils.hpp>
#include <qp/dense/precond/ruiz.hpp>
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
template <typename T, ldlt::Layout L>
using MatRef = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)> const>;
template <typename T, ldlt::Layout L>
using MatRefMut = Eigen::Ref<
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(L)>>;

template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1> const>;
template <typename T>
using VecRefMut = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T, ldlt::Layout L>
void QPupdateMatrice( //
		MatRef<T, L> H,
		MatRef<T, L> A,
		MatRef<T, L> C,
		const qp::Settings<T>& QPSettings,
		qp::dense::Data<T>& qpmodel,
		qp::dense::Workspace<T>& qpwork,
		qp::Results<T>& qpresults,
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

	qp::dense::QpViewBoxMut<T> qp_scaled{
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
		qpresults.reset_results();
	}

	// perform warm start

	qpwork.primal_feasibility_rhs_1_eq = qp::dense::infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = qp::dense::infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = qp::dense::infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = qp::dense::infty_norm(qpmodel.g);
	qpwork.correction_guess_rhs_g = qp::dense::infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			qpresults.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-qpresults.mu_eq_inv); // mu stores the inverse of mu

	qpwork.ldl.factorize(qpwork.kkt, stack);

	qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
	qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;

	qp::dense::iterative_solve_with_permut_fact( //
			QPSettings,
			qpmodel,
			qpresults,
			qpwork,
			T(1),
			qpmodel.dim + qpmodel.n_eq);

	qpresults.x = qpwork.dw_aug.head(qpmodel.dim);
	qpresults.y = qpwork.dw_aug.segment(qpmodel.dim, qpmodel.n_eq);

	qpwork.dw_aug.setZero();
}

template <typename T>
void QPupdateVectors( //
		VecRef<T> g,
		VecRef<T> b,
		VecRef<T> u,
		VecRef<T> l,
		const qp::Settings<T>& QPSettings,
		qp::dense::Data<T>& qpmodel,
		qp::dense::Workspace<T>& qpwork,
		qp::Results<T>& qpresults,
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
			qp::dense::VectorViewMut<T>{from_eigen, qpwork.g_scaled});
	qpwork._scale_dual_in_place_eq(qp::dense::VectorViewMut<T>{from_eigen, qpwork.b_scaled});
	qpwork._scale_dual_in_place_in(qp::dense::VectorViewMut<T>{from_eigen, qpwork.u_scaled});
	qpwork._scale_dual_in_place_in(qp::dense::VectorViewMut<T>{from_eigen, qpwork.l_scaled});

}

template <typename T, ldlt::Layout L>
void QPsolve(
		const qp::dense::Data<T>& qpmodel,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork,
		const qp::Settings<T>& qpsettings) {

	auto start = std::chrono::high_resolution_clock::now();
	qp::dense::qp_solve( //
			qpsettings,
			qpmodel,
			qpresults,
			qpwork);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	qpresults.timing = duration.count();

	if (qpsettings.verbose) {
		std::cout << "------ SOLVER STATISTICS--------" << std::endl;
		std::cout << "n_ext : " << qpresults.n_ext << std::endl;
		std::cout << "n_tot : " << qpresults.n_tot << std::endl;
		std::cout << "mu updates : " << qpresults.n_mu_change << std::endl;
		std::cout << "objValue : " << qpresults.objValue << std::endl;
		std::cout << "timing : " << qpresults.timing << std::endl;
	}
}

template <typename T>
void QPreset(
		const qp::dense::Data<T>& qpmodel,
		const qp::Settings<T>& qpsettings,
		qp::Results<T>& qpresults,
		qp::dense::Workspace<T>& qpwork) {

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

		qp::dense::iterative_solve_with_permut_fact( //
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
	::pybind11::class_<qp::dense::Workspace<f64>>(m, "Workspace")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member
			.def_readwrite("H_scaled", &qp::dense::Workspace<f64>::H_scaled)
			.def_readwrite("g_scaled", &qp::dense::Workspace<f64>::g_scaled)
			.def_readwrite("A_scaled", &qp::dense::Workspace<f64>::A_scaled)
			.def_readwrite("C_scaled", &qp::dense::Workspace<f64>::C_scaled)
			.def_readwrite("b_scaled", &qp::dense::Workspace<f64>::b_scaled)
			.def_readwrite("u_scaled", &qp::dense::Workspace<f64>::u_scaled)
			.def_readwrite("l_scaled", &qp::dense::Workspace<f64>::l_scaled)
			.def_readwrite("x_prev", &qp::dense::Workspace<f64>::x_prev)
			.def_readwrite("y_prev", &qp::dense::Workspace<f64>::y_prev)
			.def_readwrite("z_prev", &qp::dense::Workspace<f64>::z_prev)
			.def_readwrite("kkt", &qp::dense::Workspace<f64>::kkt)
			.def_readwrite(
					"current_bijection_map", &qp::dense::Workspace<f64>::current_bijection_map)
			.def_readwrite(
					"new_bijection_map", &qp::dense::Workspace<f64>::new_bijection_map)
			.def_readwrite("active_set_up", &qp::dense::Workspace<f64>::active_set_up)
			.def_readwrite("active_set_low", &qp::dense::Workspace<f64>::active_set_low)
			.def_readwrite(
					"active_inequalities", &qp::dense::Workspace<f64>::active_inequalities)
			.def_readwrite("Hdx", &qp::dense::Workspace<f64>::Hdx)
			.def_readwrite("Cdx", &qp::dense::Workspace<f64>::Cdx)
			.def_readwrite("Adx", &qp::dense::Workspace<f64>::Adx)
			.def_readwrite("active_part_z", &qp::dense::Workspace<f64>::active_part_z)
			.def_readwrite("alphas", &qp::dense::Workspace<f64>::alphas)
			.def_readwrite("dw_aug", &qp::dense::Workspace<f64>::dw_aug)
			.def_readwrite("rhs", &qp::dense::Workspace<f64>::rhs)
			.def_readwrite("err", &qp::dense::Workspace<f64>::err)
			.def_readwrite(
					"primal_feasibility_rhs_1_eq",
					&qp::dense::Workspace<f64>::primal_feasibility_rhs_1_eq)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_u",
					&qp::dense::Workspace<f64>::primal_feasibility_rhs_1_in_u)
			.def_readwrite(
					"primal_feasibility_rhs_1_in_l",
					&qp::dense::Workspace<f64>::primal_feasibility_rhs_1_in_l)
			.def_readwrite(
					"dual_feasibility_rhs_2",
					&qp::dense::Workspace<f64>::dual_feasibility_rhs_2)
			.def_readwrite(
					"correction_guess_rhs_g",
					&qp::dense::Workspace<f64>::correction_guess_rhs_g)
			.def_readwrite("alpha", &qp::dense::Workspace<f64>::alpha)
			.def_readwrite(
					"dual_residual_scaled", &qp::dense::Workspace<f64>::dual_residual_scaled)
			.def_readwrite(
					"primal_residual_eq_scaled",
					&qp::dense::Workspace<f64>::primal_residual_eq_scaled)
			.def_readwrite(
					"primal_residual_in_scaled_up",
					&qp::dense::Workspace<f64>::primal_residual_in_scaled_up)
			.def_readwrite(
					"primal_residual_in_scaled_low",
					&qp::dense::Workspace<f64>::primal_residual_in_scaled_low)
			.def_readwrite(
					"primal_residual_in_scaled_up_plus_alphaCdx",
					&qp::dense::Workspace<f64>::primal_residual_in_scaled_up_plus_alphaCdx)
			.def_readwrite(
					"primal_residual_in_scaled_low_plus_alphaCdx",
					&qp::dense::Workspace<f64>::primal_residual_in_scaled_low_plus_alphaCdx)
			.def_readwrite("CTz", &qp::dense::Workspace<f64>::CTz);

	::pybind11::class_<qp::Results<f64>>(m, "Results")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member

			.def_readwrite("x", &qp::Results<f64>::x)
			.def_readwrite("y", &qp::Results<f64>::y)
			.def_readwrite("z", &qp::Results<f64>::z)
			.def_readwrite("n_c", &qp::Results<f64>::n_c)
			.def_readwrite("mu_eq", &qp::Results<f64>::mu_eq)
			.def_readwrite("mu_eq_inv", &qp::Results<f64>::mu_eq_inv)
			.def_readwrite("mu_in", &qp::Results<f64>::mu_in)
			.def_readwrite("mu_in_inv", &qp::Results<f64>::mu_in_inv)
			.def_readwrite("rho", &qp::Results<f64>::rho)
			.def_readwrite("n_tot", &qp::Results<f64>::n_tot)
			.def_readwrite("n_ext", &qp::Results<f64>::n_ext)
			.def_readwrite("timing", &qp::Results<f64>::timing)
			.def_readwrite("objValue", &qp::Results<f64>::objValue)
			.def_readwrite("n_mu_change", &qp::Results<f64>::n_mu_change);

	::pybind11::class_<qp::Settings<f64>>(m, "Settings")
			.def(::pybind11::init()) // constructor
	                             // read-write public data member

			.def_readwrite("alpha_bcl", &qp::Settings<f64>::alpha_bcl)
			.def_readwrite("beta_bcl", &qp::Settings<f64>::beta_bcl)
			.def_readwrite(
					"refactor_dual_feasibility_threshold",
					&qp::Settings<f64>::refactor_dual_feasibility_threshold)
			.def_readwrite(
					"refactor_rho_threshold",
					&qp::Settings<f64>::refactor_rho_threshold)
			.def_readwrite("mu_max_eq", &qp::Settings<f64>::mu_max_eq)
			.def_readwrite("mu_max_in", &qp::Settings<f64>::mu_max_in)
			.def_readwrite("mu_max_eq_inv", &qp::Settings<f64>::mu_max_eq_inv)
			.def_readwrite("mu_max_in_inv", &qp::Settings<f64>::mu_max_in_inv)
			.def_readwrite("mu_update_factor", &qp::Settings<f64>::mu_update_factor)
			.def_readwrite(
					"mu_update_inv_factor", &qp::Settings<f64>::mu_update_inv_factor)

			.def_readwrite("cold_reset_mu_eq", &qp::Settings<f64>::cold_reset_mu_eq)
			.def_readwrite("cold_reset_mu_in", &qp::Settings<f64>::cold_reset_mu_in)
			.def_readwrite(
					"cold_reset_mu_eq_inv", &qp::Settings<f64>::cold_reset_mu_eq_inv)

			.def_readwrite(
					"cold_reset_mu_in_inv", &qp::Settings<f64>::cold_reset_mu_in_inv)
			.def_readwrite("max_iter", &qp::Settings<f64>::max_iter)
			.def_readwrite("max_iter_in", &qp::Settings<f64>::max_iter_in)

			.def_readwrite("eps_abs", &qp::Settings<f64>::eps_abs)
			.def_readwrite("eps_rel", &qp::Settings<f64>::eps_rel)
			.def_readwrite("eps_IG", &qp::Settings<f64>::eps_IG)
			.def_readwrite("R", &qp::Settings<f64>::R)
			.def_readwrite(
					"nb_iterative_refinement",
					&qp::Settings<f64>::nb_iterative_refinement)
			.def_readwrite("warm_start", &qp::Settings<f64>::warm_start)
			.def_readwrite("verbose", &qp::Settings<f64>::verbose);

	::pybind11::class_<qp::dense::Data<f64>>(m, "Data")
			.def(::pybind11::init<i64, i64, i64&>()) // constructor
	                                             // read-write public data member

			.def_readonly("H", &qp::dense::Data<f64>::H)
			.def_readonly("g", &qp::dense::Data<f64>::g)
			.def_readonly("A", &qp::dense::Data<f64>::A)
			.def_readonly("b", &qp::dense::Data<f64>::b)
			.def_readonly("C", &qp::dense::Data<f64>::C)
			.def_readonly("u", &qp::dense::Data<f64>::u)
			.def_readonly("l", &qp::dense::Data<f64>::l)
			.def_readonly("dim", &qp::dense::Data<f64>::dim)
			.def_readonly("n_eq", &qp::dense::Data<f64>::n_eq)
			.def_readonly("n_in", &qp::dense::Data<f64>::n_in)
			.def_readonly("n_total", &qp::dense::Data<f64>::n_total);

	constexpr auto c = rowmajor;

	m.def("QPsolve", &qp::pybind11::QPsolve<f32, c>);
	m.def("QPsolve", &qp::pybind11::QPsolve<f64, c>);

	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f32, c>);
	m.def("QPupdateMatrice", &qp::pybind11::QPupdateMatrice<f64, c>);

	m.def("QPsetup", &qp::dense::QPsetup<f32>);
	m.def("QPsetup", &qp::dense::QPsetup<f64>);

	m.def("QPreset", &qp::pybind11::QPreset<f32>);
	m.def("QPreset", &qp::pybind11::QPreset<f64>);

	m.attr("__version__") = "dev";
}
