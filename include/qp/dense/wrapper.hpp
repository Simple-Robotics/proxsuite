#ifndef PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP
#include <qp/Results.hpp>
#include <qp/Settings.hpp>
#include <qp/dense/solver.hpp>
#include <chrono>

namespace qp{
namespace dense {

static constexpr auto DYN = Eigen::Dynamic;
enum { layout = Eigen::RowMajor };
template <typename T>
using SparseMat = Eigen::SparseMatrix<T, 1>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef =Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Mat = Eigen::Matrix<T, DYN, DYN, layout>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

/////// SETUP ////////
template <typename T>
void initial_guess(dense::Workspace<T>& qpwork,
                   Settings<T>& qpsettings,
                   dense::Data<T>& qpmodel,
                   Results<T>& qpresults){
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

	qpwork.primal_feasibility_rhs_1_eq = dense::infty_norm(qpmodel.b);
	qpwork.primal_feasibility_rhs_1_in_u = dense::infty_norm(qpmodel.u);
	qpwork.primal_feasibility_rhs_1_in_l = dense::infty_norm(qpmodel.l);
	qpwork.dual_feasibility_rhs_2 = dense::infty_norm(qpmodel.g);
    qpwork.correction_guess_rhs_g = qp::dense::infty_norm(qpwork.g_scaled);

	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim) = qpwork.H_scaled;
	qpwork.kkt.topLeftCorner(qpmodel.dim, qpmodel.dim).diagonal().array() +=
			qpresults.info.rho;
	qpwork.kkt.block(0, qpmodel.dim, qpmodel.dim, qpmodel.n_eq) =
			qpwork.A_scaled.transpose();
	qpwork.kkt.block(qpmodel.dim, 0, qpmodel.n_eq, qpmodel.dim) = qpwork.A_scaled;
	qpwork.kkt.bottomRightCorner(qpmodel.n_eq, qpmodel.n_eq).setZero();
	qpwork.kkt.diagonal()
			.segment(qpmodel.dim, qpmodel.n_eq)
			.setConstant(-qpresults.info.mu_eq);

	qpwork.ldl.factorize(qpwork.kkt, stack);

	if (!qpsettings.warm_start) {
		qpwork.rhs.head(qpmodel.dim) = -qpwork.g_scaled;
		qpwork.rhs.segment(qpmodel.dim, qpmodel.n_eq) = qpwork.b_scaled;
		iterative_solve_with_permut_fact( //
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
};

template <typename Mat, typename T>
void setup_generic( //
		Mat const& H,
		VecRef<T> g,
		Mat const& A,
		VecRef<T> b,
		Mat const& C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults) {

	auto start = std::chrono::steady_clock::now();
	qpmodel.H = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(H);
	qpmodel.g = g;
	qpmodel.A = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(A);
	qpmodel.b = b;
	qpmodel.C = Eigen::
			Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(C);
	qpmodel.u = u;
	qpmodel.l = l;

	qpwork.H_scaled = qpmodel.H;
	qpwork.g_scaled = qpmodel.g;
	qpwork.A_scaled = qpmodel.A;
	qpwork.b_scaled = qpmodel.b;
	qpwork.C_scaled = qpmodel.C;
	qpwork.u_scaled = qpmodel.u;
	qpwork.l_scaled = qpmodel.l;

    initial_guess(qpwork,qpsettings,qpmodel,qpresults);

	auto stop = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	qpresults.info.setup_time = duration.count();
}

template <typename T>
void setup_dense( //
		MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults
) {
	setup_generic(
			H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

template <typename T>
void setup_sparse( //
		const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l,
		Settings<T>& qpsettings,
		dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
		Results<T>& qpresults) {
	setup_generic(H, g, A, b, C, u, l, qpsettings, qpmodel, qpwork, qpresults);
}

////// UPDATES ///////
template <typename T>
void update_lin_cost(
        dense::Data<T>& qpmodel,
		dense::Workspace<T>& qpwork,
        VecRef<T> g_){
    qpmodel.g = g_.eval();
    qpwork.g_scaled = qpmodel.g;
    qpwork.ruiz.scale_primal_in_place(VectorViewMut<T>{from_eigen, qpwork.g_scaled});
};
template <typename T>
void update_lower_bound(
            dense::Data<T>& qpmodel,
		    dense::Workspace<T>& qpwork,
            VecRef<T> l_){
    qpmodel.l = l_.eval();
    qpwork.l_scaled = qpmodel.l;
    qpwork.ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.l_scaled});
};
template <typename T>
void update_upper_bound(
            dense::Data<T>& qpmodel,
		    dense::Workspace<T>& qpwork,
            VecRef<T> u_){
    qpmodel.u = u_.eval();
    qpwork.u_scaled = qpmodel.u;
    qpwork.ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.u_scaled});
};
template <typename T>
void update_equality_bound(
            dense::Data<T>& qpmodel,
		    dense::Workspace<T>& qpwork,
            VecRef<T> b_){
    qpmodel.b = b_.eval();
    qpwork.b_scaled = qpmodel.b;
    qpwork.ruiz.scale_dual_in_place_in(VectorViewMut<T>{from_eigen, qpwork.b_scaled});
};
template <typename T>
void update_matrices(dense::Data<T>& qpmodel,
		    dense::Workspace<T>& qpwork,
            Settings<T>& qpsettings,
            Results<T>& qpresults,
            MatRef<T> H_,
            MatRef<T> A_,
            MatRef<T> C_){
    // TODO: use std::optional for matrices argument
    qpmodel.H = H_.eval();
    qpmodel.A = A_.eval();
    qpmodel.C = C_.eval();
    qpwork.H_scaled = qpmodel.H;
    qpwork.A_scaled = qpmodel.A;
    qpwork.C_scaled = qpmodel.C;

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
	qpresults.reset_results(); // re update all other variables
    initial_guess(qpwork,qpsettings,qpmodel,qpresults);
};

template <typename T>
void update_proximal_parameters(Results<T>& results,Workspace<T>& work, Settings<T>& settings, Data<T>& qpmodel, T rho_new, T mu_eq_new, T mu_in_new){
    // TODO: use std::optional for matrices argument
    results.info.rho = rho_new;
    results.info.mu_eq = mu_eq_new;
    results.info.mu_eq_inv = T(1)/mu_eq_new;
    results.info.mu_in = mu_in_new;
    results.info.mu_in_inv = T(1)/mu_in_new;
    initial_guess(work,settings,qpmodel,results);
};
template<typename T>
void warm_starting(VecRef<T> x_wm,
               VecRef<T> y_wm,
               VecRef<T> z_wm,
               Results<T>& results){
    // TODO: use std::optional for matrices argument
    results.x = x_wm.eval();
    results.y = y_wm.eval();
    results.z = z_wm.eval();
};

///// QP object
template <typename T>
struct QP {
public:
    
    Results<T> results; 
    Settings<T> settings;
    Data<T> data;
    Workspace<T> work;
    
    QP(isize _dim, isize _n_eq, isize _n_in):data(_dim, _n_eq, _n_in),work(_dim, _n_eq, _n_in),settings(),results(_dim, _n_eq, _n_in){
    }

    void setup_dense_matrices(MatRef<T> H,
		VecRef<T> g,
		MatRef<T> A,
		VecRef<T> b,
		MatRef<T> C,
		VecRef<T> u,
		VecRef<T> l){
            setup_dense(H,g,A,b,C,u,l,settings,data,work,results);
        };
    void setup_sparse_matrices(const SparseMat<T>& H,
		VecRef<T> g,
		const SparseMat<T>& A,
		VecRef<T> b,
		const SparseMat<T>& C,
		VecRef<T> u,
		VecRef<T> l){
            setup_sparse(H,g,A,b,C,u,l,settings,data,work,results);
        };

    void solve(){

        auto start = std::chrono::high_resolution_clock::now();
        qp_solve( //
                settings,
                data,
                results,
                work);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        results.info.solve_time = duration.count();
        results.info.run_time =
                results.info.solve_time + results.info.setup_time;

        if (settings.verbose) {
            std::cout << "------ SOLVER STATISTICS--------" << std::endl;
            std::cout << "iter_ext : " << results.info.iter_ext << std::endl;
            std::cout << "iter : " << results.info.iter << std::endl;
            std::cout << "mu updates : " << results.info.mu_updates << std::endl;
            std::cout << "rho_updates : " << results.info.rho_updates << std::endl;
            std::cout << "objValue : " << results.info.objValue << std::endl;
            std::cout << "solve_time : " << results.info.solve_time << std::endl;
        }
    };

    void update(MatRef<T> H_, VecRef<T> g_, MatRef<T> A_, VecRef<T> b_, MatRef<T> C_, VecRef<T> u_, VecRef<T> l_){
        // TODO use std optional 
        update_matrices(data, work, settings,results, H_, A_, C_);
        update_lin_cost(data,work,g_);
        update_equality_bound(data,work,b_);
        update_upper_bound(data,work,u_);
        update_lower_bound(data,work,l_);
    }
    void update_prox_parameter(T rho_new, T mu_eq_new, T mu_in_new){
        // TODO use std optional 
        update_proximal_parameters(results,work,settings,data,rho_new, mu_eq_new, mu_in_new);
    };
    void warm_sart(VecRef<T> x_wm,
               VecRef<T> y_wm,
               VecRef<T> z_wm){
        warm_starting(x_wm,y_wm,z_wm,results);
    };
    void cleanup(){
        results.reset_results();
        work.reset_workspace();
        initial_guess(work,settings,data,results);
    }
};


} // namespace dense
} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_WRAPPER_HPP */