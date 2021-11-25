#include <qp/proxqp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>

#include <numeric>
#include <util.hpp>
#include <qp/views.hpp>
#include <qp/QPWorkspace.hpp>
#include <qp/QPData.hpp>
#include <qp/QPResults.hpp>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

using Scalar = double;

auto main() -> int {
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using ldlt::i64;
	using ldlt::isize;
	isize dim = 50;
    isize n_eq = 0;
    isize n_in = isize(dim/2);
    double sparsity_factor(0.5);

	isize max_iter(100);
    isize max_iter_in(2500);
    Scalar eps_abs = Scalar(1e-9);
    Scalar eps_rel = 0;
    Scalar err_IG = 1.e-4;
    Scalar beta = 0.9 ;
    Scalar R = 3 ;
	Qp<Scalar> qp_problem{random_with_dim_and_n_in, dim, sparsity_factor};


    bool VERBOSE = false;
    Vec x = Vec::Zero(dim);
	Vec y = Vec::Zero(n_eq);
    Vec z = Vec::Zero(n_in);

    // allocating all needed variables
    
    qp::Qpdata<Scalar> qpmodel{qp_problem.H,
                               qp_problem.g,
                               qp_problem.A,
                               qp_problem.b,
                               qp_problem.C,
                               qp_problem.u,
                               qp_problem.l};
    
    /*                         
    auto qpview = qp::QpViewBox<Scalar>{
					{ldlt::from_eigen, qp_problem.H},
					{ldlt::from_eigen, qp_problem.g},
					{ldlt::from_eigen, qp_problem.A},
					{ldlt::from_eigen, qp_problem.b},
					{ldlt::from_eigen, qp_problem.C}, 
					{ldlt::from_eigen, qp_problem.u},
					{ldlt::from_eigen, qp_problem.l},
		};
    */
    
    auto x_view = ldlt::VectorViewMut<Scalar>{ldlt::from_eigen, x};
    auto y_view = ldlt::VectorViewMut<Scalar>{ldlt::from_eigen, y};
    auto z_view = ldlt::VectorViewMut<Scalar> {ldlt::from_eigen, z};

    using namespace std::chrono;
    #ifndef NDEBUG
    isize n_iter(0);
    #else
    isize n_iter(10000);
    #endif
    qp::Qpworkspace<Scalar> qpwork{dim, n_eq, n_in};
    
    qp::Qpresults<Scalar> qpresults{dim,n_eq,n_in};

    
    auto start = high_resolution_clock::now();
    qp::detail::qpSolve( //
            qpmodel,
            qpwork,
            qpresults,
            max_iter,
            max_iter_in,
            eps_abs,
            eps_rel,
            err_IG,
            beta,
            R,
            VERBOSE);
    
    for (isize i=0;i<n_iter;i++){
        qpresults.clearResults();
        qp::detail::qpSolve( //
            qpmodel,
            qpwork,
            qpresults,
            max_iter,
            max_iter_in,
            eps_abs,
            eps_rel,
            err_IG,
            beta,
            R,
            VERBOSE);
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "n : " << dim << " n_in " << n_in << std::endl;
    std::cout << "Average time taken : " << duration.count()/ (n_iter+1) << " microseconds" << std::endl;
    std::cout << "n_ext : " << qpresults._n_ext << std::endl;
	std::cout << "n_tot : " << qpresults._n_tot << std::endl;
	std::cout << "mu updates : " << qpresults._n_mu_change << std::endl;

    Vec Cx = qp_problem.C * qpresults._x ;
    Vec Ax = qp_problem.A * qpresults._x  - qp_problem.b;

    Vec pri_res =  qp::detail::positive_part(Cx - qp_problem.u) + qp::detail::negative_part(Cx - qp_problem.l);

    std::cout << "primal residual : " <<  pri_res.template lpNorm<Eigen::Infinity>() << std::endl;

    Vec dua_res = qp_problem.H * qpresults._x  + qp_problem.g + qp_problem.A.transpose() * qpresults._y + qp_problem.C.transpose()* qpresults._z;

    std::cout << "dual residual : " <<  dua_res.template lpNorm<Eigen::Infinity>() << std::endl;
    
}
