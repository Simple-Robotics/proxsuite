#include <qp/proxqp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>

#include <numeric>
#include <util.hpp>
#include <qp/views.hpp>
#include <qp/QPWorkspace.hpp>
#include <qp/QPData.hpp>
#include <qp/QPResults.hpp>
#include <qp/QPSettings.hpp>

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

	Qp<Scalar> qp_problem{random_with_dim_and_n_in, dim, sparsity_factor};


    bool VERBOSE = false;
    Vec x = Vec::Zero(dim);
	Vec y = Vec::Zero(n_eq);
    Vec z = Vec::Zero(n_in);

    // allocating all needed variables 

    qp::Qpsettings<Scalar> qpsettings{};
    
    qp::Qpdata<Scalar> qpmodel{qp_problem.H,
                               qp_problem.g,
                               qp_problem.A,
                               qp_problem.b,
                               qp_problem.C,
                               qp_problem.u,
                               qp_problem.l};
    
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
            qpsettings);
    
    for (isize i=0;i<n_iter;i++){
        qpresults.clearResults();
        qp::detail::qpSolve( //
            qpmodel,
            qpwork,
            qpresults,
            qpsettings);
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
