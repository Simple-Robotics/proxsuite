#include <qp/proxqp/in_solver.hpp>
#include <qp/precond/ruiz.hpp>

#include <numeric>
#include <util.hpp>
#include <qp/views.hpp>
#include <qp/utils.hpp>

#include <fmt/chrono.h>
#include <fmt/ranges.h>

using Scalar = double;

auto main() -> int {
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using ldlt::i64;
	using ldlt::isize;
	isize dim = 1000;
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
	Qp<Scalar> qp{random_with_dim_and_n_in, dim, sparsity_factor};
    bool VERBOSE = false;
    Vec x = Vec::Zero(dim);
	Vec y = Vec::Zero(n_eq);
    Vec z = Vec::Zero(n_in);

    auto ruiz = qp::preconditioner::RuizEquilibration<Scalar>{
			dim,
			n_eq + n_in,
	};

    /*
    std::cout << " H " << qp.H << std::endl;
    std::cout << " g " << qp.g << std::endl;
    std::cout << " A " << qp.A << std::endl;
    std::cout << " b " << qp.b << std::endl;
    std::cout << " C " << qp.C << std::endl;
    std::cout << " u " << qp.u << std::endl;
    std::cout << " l " << qp.l << std::endl;
    */
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    qp::detail::QpSolveStats res = qp::detail::qpSolve( //
        {ldlt::from_eigen, x},
        {ldlt::from_eigen, y},
        {ldlt::from_eigen, z},
        qp::QpViewBox<Scalar>{
					{ldlt::from_eigen, qp.H.eval()},
					{ldlt::from_eigen, qp.g.eval()},
					{ldlt::from_eigen, qp.A.eval()},
					{ldlt::from_eigen, qp.b.eval()},
					{ldlt::from_eigen, qp.C.eval()},
					{ldlt::from_eigen, qp.u.eval()},
					{ldlt::from_eigen, qp.l.eval()},
		},
        max_iter,
        max_iter_in,
        eps_abs,
        eps_rel,
        err_IG,
        beta,
        R,
        LDLT_FWD(ruiz),
        VERBOSE);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "n : " << dim << " n_in " << n_in << std::endl;
    std::cout << "Time taken : " << duration.count() << " microseconds" << std::endl;
    std::cout << "n_ext : " << res.n_ext << std::endl;
	std::cout << "n_tot : " << res.n_tot << std::endl;
	std::cout << "mu updates : " << res.n_mu_updates << std::endl;

    Vec Cx = qp.C * x ;
    Vec Ax = qp.A * x - qp.b;

    Vec pri_res =  qp::detail::positive_part(Cx - qp.u) + qp::detail::negative_part(Cx - qp.l);

    std::cout << "primal residual : " <<  pri_res.template lpNorm<Eigen::Infinity>() << std::endl;

    Vec dua_res = qp.H * x + qp.g + qp.A.transpose() * y + qp.C.transpose()* z;

    std::cout << "dual residual : " <<  dua_res.template lpNorm<Eigen::Infinity>() << std::endl;
    
}
