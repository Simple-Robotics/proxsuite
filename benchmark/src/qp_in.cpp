#include <qp/proxqp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>

#include <numeric>
#include <util.hpp>
#include <qp/views.hpp>
#include <qp/utils.hpp>
#include <qp/QPData.hpp>

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

	isize max_iter(1000);
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


    auto ruiz = qp::preconditioner::RuizEquilibration<Scalar>{
			dim,
			n_eq + n_in,
	};
    auto qpview = qp::QpViewBox<Scalar>{
					{ldlt::from_eigen, qp_problem.H.eval()},
					{ldlt::from_eigen, qp_problem.g.eval()},
					{ldlt::from_eigen, qp_problem.A.eval()},
					{ldlt::from_eigen, qp_problem.b.eval()},
					{ldlt::from_eigen, qp_problem.C.eval()},
					{ldlt::from_eigen, qp_problem.u.eval()},
					{ldlt::from_eigen, qp_problem.l.eval()},
		};
    auto x_view = ldlt::VectorViewMut<Scalar>{ldlt::from_eigen, x};
    auto y_view = ldlt::VectorViewMut<Scalar>{ldlt::from_eigen, y};
    auto z_view = ldlt::VectorViewMut<Scalar> {ldlt::from_eigen, z};

    using namespace std::chrono;
    qp::detail::QpSolveStats res;
    auto start = high_resolution_clock::now();
    isize n_iter(1000);
    qp::Qpdata<Scalar> qpdata{
            dim, n_eq, n_in
        };
    res= qp::detail::qpSolve( //
            x_view,
            y_view,
            z_view,
            qpview,
            qpdata,
            max_iter,
            max_iter_in,
            eps_abs,
            eps_rel,
            err_IG,
            beta,
            R,
            LDLT_FWD(ruiz),
            VERBOSE);
    for (isize i=0;i<n_iter;i++){
        qp::Qpdata<Scalar> qpdata2{
            dim, n_eq, n_in
        };
        res= qp::detail::qpSolve( //
            x_view,
            y_view,
            z_view,
            qpview,
            qpdata2,
            max_iter,
            max_iter_in,
            eps_abs,
            eps_rel,
            err_IG,
            beta,
            R,
            LDLT_FWD(ruiz),
            VERBOSE);

    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "n : " << dim << " n_in " << n_in << std::endl;
    std::cout << "Average time taken : " << duration.count()/n_iter << " microseconds" << std::endl;
    std::cout << "n_ext : " << res.n_ext << std::endl;
	std::cout << "n_tot : " << res.n_tot << std::endl;
	std::cout << "mu updates : " << res.n_mu_updates << std::endl;

    Vec Cx = qp_problem.C * qpdata._x ;
    Vec Ax = qp_problem.A * qpdata._x  - qp_problem.b;

    Vec pri_res =  qp::detail::positive_part(Cx - qp_problem.u) + qp::detail::negative_part(Cx - qp_problem.l);

    std::cout << "primal residual : " <<  pri_res.template lpNorm<Eigen::Infinity>() << std::endl;

    Vec dua_res = qp_problem.H * qpdata._x  + qp_problem.g + qp_problem.A.transpose() * qpdata._y + qp_problem.C.transpose()* qpdata._z;

    std::cout << "dual residual : " <<  dua_res.template lpNorm<Eigen::Infinity>() << std::endl;

}
