#ifndef INRIA_LDLT_TEST_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_TEST_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/line_search.hpp"
#include <cmath>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

template <typename T>
void test(
		qp::QpViewBox<T> qp
        ) {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
    isize n_in = qp.C.rows;
	isize n_c = 0 ;

	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto mu_eq = T(1e3);
    auto mu_in = T(1e1);
	
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_h_scaled,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T),
	     	(_g_scaled,Init, Vec(dim),LDLT_CACHELINE_BYTES, T),
	     	(_a_scaled,Uninit, Mat(n_eq, dim),LDLT_CACHELINE_BYTES, T),
		 	(_c_scaled,Uninit, Mat(n_in, dim),LDLT_CACHELINE_BYTES, T),
	     	(_b_scaled,Init, Vec(n_eq),LDLT_CACHELINE_BYTES, T),
         	(_u_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
         	(_l_scaled,Init, Vec(n_in),LDLT_CACHELINE_BYTES, T),
			(_kkt,Uninit, Mat(dim+n_eq, dim+n_eq),LDLT_CACHELINE_BYTES, T),
            (row_,Init, Vec(n_c+1+n_eq+dim),LDLT_CACHELINE_BYTES, T)
		);

	auto H_copy = _h_scaled.to_eigen();
	auto kkt = _kkt.to_eigen();
	auto q_copy = _g_scaled.to_eigen();
	auto A_copy = _a_scaled.to_eigen();
	auto C_copy = _c_scaled.to_eigen();
	auto b_copy = _b_scaled.to_eigen();
    auto u_copy = _u_scaled.to_eigen();
    auto l_copy = _l_scaled.to_eigen();

    H_copy = qp.H.to_eigen();
    A_copy = qp.A.to_eigen();
    C_copy = qp.C.to_eigen();
    q_copy = qp.g.to_eigen();
    b_copy = qp.b.to_eigen();
    l_copy = qp.l.to_eigen();
    u_copy = qp.u.to_eigen();


    auto row = row_.to_eigen();

	auto qp_scaled = qp::QpViewBoxMut<T>{
			{from_eigen, H_copy},
			{from_eigen, q_copy},
			{from_eigen, A_copy},
			{from_eigen, b_copy},
			{from_eigen, C_copy},
			{from_eigen, u_copy},
            {from_eigen, l_copy}
	};

	kkt.topLeftCorner(dim, dim) = qp_scaled.H.to_eigen();
	for (isize i = 0; i < dim; ++i) {
		kkt(i, i) += rho;
	}
	kkt.block(0,dim,dim,n_eq) = qp_scaled.A.to_eigen().transpose();
	kkt.block(dim,0,n_eq,dim) = qp_scaled.A.to_eigen();
	kkt.bottomRightCorner(n_eq+n_c, n_eq+n_c).setZero();
	{
		T tmp_eq = -T(1) / mu_eq;
		T tmp_in = -T(1) / mu_in;
		for (isize i = 0; i < n_eq; ++i) {
			kkt(dim + i, dim + i) = tmp_eq;
		}
	}
	
	ldlt::Ldlt<T> ldl{decompose, kkt};

    std::cout << "initial difference " << kkt - ldl.reconstructed_matrix() << std::endl;

    isize i_to_insert = isize(12);

    auto C_ = qp_scaled.C.to_eigen();
    std::cout << "C_.row(i) " << C_.row(i_to_insert) << std::endl;
    row.topRows(dim) = C_.row(i_to_insert);
    row.tail(n_eq+n_c+1).setZero();
    row(dim+n_eq+n_c) = -T(1)/mu_in;
    std::cout << " added row " << row << std::endl;
    ldl.insert_at(n_eq+dim+n_c, row);


    LDLT_MULTI_WORKSPACE_MEMORY(
					(_htot,Uninit, Mat(dim+n_eq+n_c+1, dim+n_eq+n_c+1),LDLT_CACHELINE_BYTES, T),
					(Htot_reconstruct_,Uninit, Mat(dim+n_eq+n_c+1, dim+n_eq+n_c+1),LDLT_CACHELINE_BYTES, T)
	);
				
    auto Htot = _htot.to_eigen().eval();
    auto Htot_reconstruct = Htot_reconstruct_.to_eigen().eval();
    Htot_reconstruct.setZero();
    Htot.setZero();

    {
            LDLT_DECL_SCOPE_TIMER("in solver", "set H", T);
            Htot.topLeftCorner(dim, dim) = qp_scaled.H.to_eigen();
            for (isize j = 0; j < dim; ++j) {
                Htot(j, j) += rho;
            }
            
            Htot.block(0,dim,dim,n_eq) = qp_scaled.A.to_eigen().transpose();
            Htot.block(dim,0,n_eq,dim) = qp_scaled.A.to_eigen();
            Htot.bottomRightCorner(n_eq+n_c, n_eq+n_c).setZero();
            {
                T tmp_eq = -T(1) / mu_eq;
                T tmp_in = -T(1) / mu_in;
                for (isize j = 0; j < n_eq; ++j) {
                    Htot(dim + j, dim + j) = tmp_eq;
                }
                Htot(dim + n_eq , dim + n_eq ) = tmp_in;

            }
            Htot.block(dim+n_eq,0,1,dim) = C_.row(i_to_insert) ;  // insert at the end
            Htot.block(0,dim+n_eq,dim,1) = C_.transpose().col(i_to_insert) ; 
    } 
    Htot_reconstruct = ldl.reconstructed_matrix() ; 
    std::cout << " Htot " << Htot << std::endl;
    std::cout << " Htot_reconstruct " << Htot_reconstruct << std::endl;
    std::cout << "diff_mat " << Htot_reconstruct -Htot << std::endl;
    std::cout << "diff_mat norm " <<  qp::infty_norm(Htot_reconstruct -Htot) << std::endl;

}



} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_TEST_SOLVER_HPP_HDWGZKCLS */
