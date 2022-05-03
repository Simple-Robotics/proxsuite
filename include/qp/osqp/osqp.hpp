#ifndef INRIA_LDLT_OSQP_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_OSQP_SOLVER_HPP_HDWGZKCLS

#include "qp/dense/views.hpp"
#include "qp/dense/utils.hpp"
#include "qp/dense/precond/identity.hpp"
#include <veg/util/dynstack_alloc.hpp>
#include <dense-ldlt/ldlt.hpp>

#include <cmath>
#include <type_traits>
#include <numeric>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace dense {

namespace detail {

struct QpSolveOSQPStats {
	isize n_ext;
	isize n_mu_updates;
	isize n_tot;
};

template <typename T>
void iterative_residual_osqp(
		QpViewBox<T> qp_scaled,
		isize dim,
		isize n_eq,
		isize n_in,
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> err_,
		T mu_eq,
		T mu_in,
		T rho) {
	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto err = err_.to_eigen();

	err = (-rhs).eval();
	err.topRows(dim) +=
			(qp_scaled.H).to_eigen() * sol.topRows(dim) + rho * sol.topRows(dim) +
			(qp_scaled.A).to_eigen().transpose() * sol.middleRows(dim, n_eq) +
			(qp_scaled.C).to_eigen().transpose() * sol.tail(n_in);
	err.middleRows(dim, n_eq) += (qp_scaled.A).to_eigen() * sol.topRows(dim) -
	                             sol.middleRows(dim, n_eq) / mu_eq;
	err.tail(n_in) +=
			(qp_scaled.C).to_eigen() * sol.topRows(dim) - sol.tail(n_in) / mu_in;
}

template <typename T>
void iterative_solve_with_permut_fact_osqp( //
		VectorViewMut<T> rhs_,
		VectorViewMut<T> sol_,
		VectorViewMut<T> res_,
		dense_ldlt::Ldlt<T>& ldl,
		T eps,
		isize max_it,
		QpViewBox<T> qp_scaled,
		isize dim,
		isize n_eq,
		isize n_in,
		T mu_eq,
		T mu_in,
		T rho,
		bool& VERBOSE) {

	auto rhs = rhs_.to_eigen();
	auto sol = sol_.to_eigen();
	auto res = res_.to_eigen();

	i32 it = 0;
	sol = rhs;
	// std::cout << "sol iterative_solve_with_permut_fact_osqp " <<  sol <<
	// std::endl;
	ldl.solve_in_place(sol);
	// std::cout << "sol after solve in place " <<  sol << std::endl;
	// std::cout << "res iterative_solve_with_permut_fact_osqp " <<  res <<
	// std::endl;
	detail::iterative_residual_osqp<T>(
			qp_scaled, dim, n_eq, n_in, rhs_, sol_, res_, mu_eq, mu_in, rho);
	// std::cout << "infty_norm(res) " << qp::infty_norm(res) << std::endl;

	while (infty_norm(res) >= eps) {
		it += 1;
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res);
		sol += res;

		res.setZero();
		detail::iterative_residual_osqp<T>(
				qp_scaled, dim, n_eq, n_in, rhs_, sol_, res_, mu_eq, mu_in, rho);
		if (VERBOSE) {
			std::cout << "infty_norm(res) " << infty_norm(res) << std::endl;
		}
	}
}

// z_eq == b
//
// [ H + rho I    AT          CT    ]       [ H×x + g + AT×y_eq + CT×y_in ]
// [ A         -1/µ_eq I      0     ]       [ A×x - z_eq                  ]
// [ C            0        -1/µ_in I] dw = -[ C×x - z_in                  ]

template <typename T>
void newton_step_osqp(
		QpViewBox<T> qp_scaled,
		VectorView<T> xe,
		VectorView<T> ye,
		VectorView<T> ze,
		VectorViewMut<T> dw_,
		VectorViewMut<T> err_,
		T mu_eq,
		T mu_in,
		T rho,
		isize dim,
		isize n_eq,
		isize n_in,
		dense_ldlt::Ldlt<T>& ldl,
		VectorViewMut<T> rhs_,
		VectorView<T> dual_residual_,      // H×x + g + AT×y_eq + CT×y_in
		VectorView<T> primal_residual_eq_, // A×x-b
		bool& VERBOSE) {

	auto rhs = rhs_.to_eigen();
	auto dual_residual = dual_residual_.to_eigen();
	auto primal_residual_eq = primal_residual_eq_.to_eigen();

	auto C_ = qp_scaled.C.to_eigen();
	auto x_e = xe.to_eigen();
	auto z_e = ze.to_eigen();
	auto dw = dw_.to_eigen();
	auto res = err_.to_eigen();
	dw.setZero();
	res.setZero();

	rhs.topRows(dim) = -dual_residual;
	rhs.middleRows(dim, n_eq) = -primal_residual_eq;
	{
		// C×x - z_in
		rhs.tail(n_in) = z_e.tail(n_in);
		rhs.tail(n_in).noalias() -= C_ * x_e;
	}

	detail::iterative_solve_with_permut_fact_osqp(
			rhs_,
			dw_,
			err_,
			ldl,
			T(1e-5),
			isize(10),
			qp_scaled,
			dim,
			n_eq,
			n_in,
			mu_eq,
			mu_in,
			rho,
			VERBOSE);
	dw_.to_eigen() = dw;
}

template <
		typename T,
		typename Preconditioner = preconditioner::IdentityPrecond>
auto osqpSolve( //
		VectorViewMut<T> xe,
		VectorViewMut<T> ye,
		QpViewBox<T> qp,
		isize max_iter,
		isize max_iter_in,
		T eps_abs,
		T eps_rel,
		Preconditioner precond = Preconditioner{},
		bool VERBOSE = false) -> QpSolveOSQPStats {

	using namespace ldlt::tags;

	isize dim = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;
	isize n_mu_updates = 0;
	isize n_tot = 0;
	isize n_ext = 0;
	isize const max_n_tot = dim + n_eq + n_in;

	T machine_eps = std::numeric_limits<T>::epsilon();
	auto rho = T(1e-6);
	auto mu_eq = T(1e4);
	auto mu_in = T(1e1);
	T alpha = T(1.6);

	auto tag = veg::Tag<T>{};
	using SR = veg::dynstack::StackReq;
	auto req = SR::and_(veg::init_list({
			dense_ldlt::temp_mat_req(tag, max_n_tot, max_n_tot), // h_tot
			dense_ldlt::temp_vec_req(tag, dim),                  // q_copy
			dense_ldlt::temp_vec_req(tag, n_eq),                 // b_copy
			dense_ldlt::temp_vec_req(tag, n_in),                 // l_copy
			dense_ldlt::temp_vec_req(tag, n_in),                 // u_copy
			dense_ldlt::temp_vec_req(tag, max_n_tot),            // residual_scaled
			dense_ldlt::temp_vec_req(tag, dim),                  // Hx
			dense_ldlt::temp_vec_req(tag, dim),                  // ATy
			dense_ldlt::temp_vec_req(tag, dim),                  // CTz
			dense_ldlt::temp_vec_req(tag, n_eq + n_in),          // ze
			dense_ldlt::temp_vec_req(tag, n_eq + n_in),          // z
			dense_ldlt::temp_vec_req(tag, max_n_tot),            // dw
			dense_ldlt::temp_vec_req(tag, max_n_tot),            // rhs
			dense_ldlt::temp_vec_req(tag, max_n_tot),            // err
			dense_ldlt::temp_vec_req(tag, n_in),                 // tmp
			dense_ldlt::Ldlt<T>::factorize_req(max_n_tot),       // ldl
	}));

	VEG_MAKE_STACK(stack, req);
	LDLT_TEMP_MAT_UNINIT(T, Htot, max_n_tot, max_n_tot, stack);
	LDLT_TEMP_VEC_UNINIT(T, q_copy, dim, stack);
	LDLT_TEMP_VEC_UNINIT(T, b_copy, n_eq, stack);
	LDLT_TEMP_VEC_UNINIT(T, l_copy, n_in, stack);
	LDLT_TEMP_VEC_UNINIT(T, u_copy, n_in, stack);
	LDLT_TEMP_VEC(T, residual_scaled, max_n_tot, stack);
	LDLT_TEMP_VEC(T, Hx, dim, stack);
	LDLT_TEMP_VEC(T, ATy, dim, stack);
	LDLT_TEMP_VEC(T, CTz, dim, stack);
	LDLT_TEMP_VEC(T, ze, n_eq + n_in, stack);
	LDLT_TEMP_VEC(T, z, n_eq + n_in, stack);
	LDLT_TEMP_VEC(T, dw, max_n_tot, stack);
	LDLT_TEMP_VEC(T, rhs, max_n_tot, stack);
	LDLT_TEMP_VEC(T, err, max_n_tot, stack);
	LDLT_TEMP_VEC(T, tmp, n_in, stack);

	q_copy = qp.g.to_eigen();
	b_copy = qp.b.to_eigen();
	u_copy = qp.u.to_eigen();
	l_copy = qp.l.to_eigen();

	dense_ldlt::Ldlt<T> ldl;
	ldl.reserve_uninit(max_n_tot);

	auto qp_scaled = [&] {
		auto qp_scaled_mut = QpViewBoxMut<T>{
				{from_eigen, Htot.topLeftCorner(dim, dim)},
				{from_eigen, q_copy},
				{from_eigen, Htot.block(dim, 0, n_eq, dim)},
				{from_eigen, b_copy},
				{from_eigen, Htot.block(dim + n_eq, 0, n_in, dim)},
				{from_eigen, u_copy},
				{from_eigen, l_copy}};
		// precond.scale_qp_in_place(qp_scaled_mut);
		return qp_scaled_mut.as_const();
	}();

	{
		Htot.bottomRightCorner(n_eq + n_in, n_eq + n_in).setZero();
		Htot.topLeftCorner(dim, dim) = qp.H.to_eigen();
		// only set bottom left half
		Htot.block(dim, 0, n_eq, dim) = qp.A.to_eigen();
		Htot.block(dim + n_eq, 0, n_in, dim) = qp.C.to_eigen();

		{
			// update diagonal H part
			for (isize i = 0; i < dim; ++i) {
				Htot(i, i) += rho;
			}

			// update diagonal constraint part
			T tmp_eq = -T(1) / mu_eq;
			T tmp_in = -T(1) / mu_in;
			for (isize i = 0; i < n_eq; ++i) {
				Htot(dim + i, dim + i) = tmp_eq;
			}
			for (isize i = 0; i < n_in; ++i) {
				Htot(dim + n_eq + i, dim + n_eq + i) = tmp_in;
			}
		}

		{
			// initial primal guess
			dense_ldlt::Ldlt<T> ldl_ws;
			ldl_ws.factorize(Htot.topLeftCorner(dim, dim));
			xe.to_eigen() = -(qp_scaled.g.to_eigen());
			ldl_ws.solve_in_place(xe.to_eigen());
		}

		ldl.factorize(Htot, stack);
	}

	ze.topRows(n_eq) = qp_scaled.b.to_eigen();
	z.topRows(n_eq) = qp_scaled.b.to_eigen();

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	auto dual_residual_scaled = residual_scaled.topRows(dim);
	auto primal_residual_eq_scaled = residual_scaled.middleRows(dim, n_eq);
	auto primal_residual_in_scaled_u = residual_scaled.bottomRows(n_in);

	T primal_feasibility_eq_rhs_0(0);
	T primal_feasibility_in_rhs_0(0);
	T dual_feasibility_rhs_0(0);
	T dual_feasibility_rhs_1(0);
	T dual_feasibility_rhs_3(0);

	T primal_feasibility_lhs(0);
	T primal_feasibility_eq_lhs(0);
	T primal_feasibility_in_lhs(0);
	T dual_feasibility_lhs(0);
	T rhs_d(0);
	T rhs_p(0);
	T fact(0);

	for (i64 iter = 0; iter <= max_iter; ++iter) {
		n_ext += 1;
		if (iter == max_iter) {
			break;
		}

		// compute primal residual
		/*
		qp::detail::global_primal_residual(
		    primal_feasibility_lhs,
		    primal_feasibility_eq_rhs_0,
		    primal_feasibility_in_rhs_0,
		    {from_eigen, primal_residual_eq_scaled},
		    {from_eigen, primal_residual_in_scaled_u},
		    {from_eigen, primal_residual_in_scaled_u},
		    qp,
		    qp_scaled,
		    precond,
		    xe.as_const());

		qp::detail::global_dual_residual(
		    dual_feasibility_lhs,
		    dual_feasibility_rhs_0,
		    dual_feasibility_rhs_1,
		    dual_feasibility_rhs_3,
		    {from_eigen, dual_residual_scaled},
		    {from_eigen, Hx},
		    {from_eigen, ATy},
		    {from_eigen, CTz},
		    {from_eigen, dw},
		    qp_scaled,
		    precond,
		    xe.as_const(),
		    ye.as_const().segment(0, n_eq),
		    ye.as_const().segment(n_eq, n_in));
		*/
		if (VERBOSE) {
			std::cout << "---------------it : " << iter
								<< " primal residual : " << primal_feasibility_lhs
								<< " dual residual : " << dual_feasibility_lhs << std::endl;
			std::cout << " rho : " << rho << " mu_eq : " << mu_eq
								<< " mu_in : " << mu_in << std::endl;
		}
		rhs_d = max_list({
				primal_feasibility_eq_rhs_0,
				primal_feasibility_in_rhs_0,
				primal_feasibility_rhs_1_eq,
				primal_feasibility_rhs_1_in_u,
				primal_feasibility_rhs_1_in_l,
		});

		rhs_p = max_list({
				dual_feasibility_rhs_3,
				dual_feasibility_rhs_0,
				dual_feasibility_rhs_1,
				dual_feasibility_rhs_2,
		});

		bool is_primal_feasible =
				primal_feasibility_lhs <= (eps_abs + eps_rel * rhs_p);

		bool is_dual_feasible = dual_feasibility_lhs <= (eps_abs + eps_rel * rhs_d);

		if (is_primal_feasible) {
			if (is_dual_feasible) {

				/*
				// POLISHING IFF IT HAS CONVERGED
				rhs.topRows(dim) = -dual_residual_scaled;
				rhs.middleRows(dim, n_eq) = -primal_residual_eq_scaled;
				{
				  LDLT_MULTI_WORKSPACE_MEMORY(
				      (test_, Init, Vec(n_in), LDLT_CACHELINE_BYTES, bool));
				  auto test = test_.to_eigen();
				  isize j(0);
				  for (isize i = 0; i < n_in; ++i) {
				    test(i) = qp_scaled.u.to_eigen()(i) - ze(i) >= ye.to_eigen()(i) &&
				              ze(i) - qp_scaled.l.to_eigen()(i) >= -ye.to_eigen()(i);
				    if (test(i)) {
				      ldl.delete_at(j + dim + n_eq);
				    } else {
				      rhs.topRows(dim) += qp_scaled.C.to_eigen().row(i) * ze(i);
				      rhs(dim + n_eq + j) =
				          -(qp_scaled.C.to_eigen().row(i).dot(xe.to_eigen()) - ze(i));
				      j += 1;
				    }
				  }
				  LDLT_MULTI_WORKSPACE_MEMORY(
				      (dw___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T),
				      (err___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T),
				      (rhs___, Init, Vec(dim + n_eq + j), LDLT_CACHELINE_BYTES, T));
				  dw.setZero();
				  err.topRows(dim + n_eq + j).setZero();
				  auto dw__ = dw___.to_eigen();
				  auto err__ = err___.to_eigen();
				  auto rhs__ = rhs___.to_eigen();
				  dw__ = dw.topRows(dim + n_eq + j);
				  err__ = err.topRows(dim + n_eq + j);
				  rhs__ = rhs.topRows(dim + n_eq + j);
				  iterative_solve_with_permut_fact_osqp(
				      rhs___,
				      dw___,
				      err___,
				      ldl,
				      T(1e-5),
				      isize(3),
				      qp_scaled,
				      dim,
				      n_eq,
				      j,
				      mu_eq,
				      mu_in,
				      rho);
				  j = isize(0);
				  for (isize i = 0; i < n_in; ++i) {
				    if (test(i)) {
				      dw(dim + n_eq + i) = dw__(dim + n_eq + j);
				      j += 1;
				    }
				  }
				}
				// see end of loop for comments
				tmp = (alpha / mu_in) * dw.tail(n_in) //
				      + ze.tail(n_in)                 //
				      + ye.to_eigen().tail(n_in) / mu_in;
				z.tail(n_in) = tmp + //
				               detail::positive_part(qp_scaled.l.to_eigen() - tmp) -
				               detail::positive_part(tmp - qp_scaled.u.to_eigen());
				ye.to_eigen().topRows(n_eq) += alpha * dw;
				ye.to_eigen().tail(n_in) = mu_in * (tmp - z.tail(n_in));
				xe.to_eigen() += alpha * dw.topRows(dim);
				// unscale polished solution
				{
				  precond.unscale_primal_in_place(xe);
				  precond.unscale_dual_in_place_eq(
				      VectorViewMut<T>{from_eigen, ye.to_eigen().topRows(n_eq)});
				  precond.unscale_dual_in_place_in(
				      VectorViewMut<T>{from_eigen, ye.to_eigen().tail(n_in)});
				}
				*/
				return {n_ext, n_mu_updates, n_tot};
			}
		}

		// mu update
		/*
		if (iter > 1 && iter % 100 == 0 ) {
		  using std::sqrt;
		  fact = sqrt(
		      (primal_feasibility_lhs * rhs_d) /
		      (dual_feasibility_lhs * rhs_p + machine_eps));
		  if (fact > T(5) || fact < T(0.2)) {
		    T mu_in_new = min2(mu_in * fact, T(1e6));
		    T mu_eq_new = min2(mu_eq * fact, T(1e6));
		    if (mu_in_new != T(1.e6) || mu_eq_new != T(1e6)) {
		      qp::detail::mu_update(
		          mu_eq, mu_eq_new, mu_in, mu_in_new, dim, n_eq, n_in, ldl);
		    }
		    mu_in = mu_in_new;
		    mu_eq = mu_eq_new;
		    n_mu_updates += 1;
		  }
		}
		*/
		// NEWTON STEP
		// std::cout << " err before newton " << err << std::endl;
		detail::newton_step_osqp(
				qp_scaled,
				xe.as_const(),
				ye.as_const(),
				VectorView<T>{from_eigen, ze},
				VectorViewMut<T>{from_eigen, dw},
				VectorViewMut<T>{from_eigen, err},
				mu_eq,
				mu_in,
				rho,
				dim,
				n_eq,
				n_in,
				ldl,
				VectorViewMut<T>{from_eigen, rhs},
				VectorView<T>{from_eigen, dual_residual_scaled},
				VectorView<T>{from_eigen, primal_residual_eq_scaled},
				VERBOSE);

		// ITERATES UPDATES according to OSQP algorithm 1 page 9 using

		// tmp = alpha/µ dw + zk + yk/µ
		tmp = (alpha / mu_in) * dw.tail(n_in) //
		      + ze.tail(n_in)                 //
		      + ye.to_eigen().tail(n_in) / mu_in;
		z.tail(n_in) = tmp + //
		               dense::positive_part(qp_scaled.l.to_eigen() - tmp) -
		               dense::positive_part(tmp - qp_scaled.u.to_eigen());
		// y{k+1} = yk + µ ( alpha (zk + dw/µ) + (1-alpha) zk - z{k+1} )
		//        = yk + µ ( zk - z{k+1} + alpha/µ dw )
		//        = yk + µ (zk - z{k+1}) + alpha dw
		//
		// eq constraints: z_k == z_{k+1} == b
		ye.to_eigen().topRows(n_eq) += alpha * dw.middleRows(dim, n_eq);
		ye.to_eigen().tail(n_in) = mu_in * (tmp - z.tail(n_in));
		xe.to_eigen() += alpha * dw.topRows(dim);
		ze = z;
	}

	return {max_iter, n_mu_updates, n_tot};
}

} // namespace detail
} // namespace dense
} // namespace qp

#endif /* end of include guard INRIA_LDLT_OSQP_SOLVER_HPP_HDWGZKCLS */
