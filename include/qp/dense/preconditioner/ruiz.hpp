/**
 * @file ruiz.hpp 
*/
#ifndef PROXSUITE_QP_DENSE_PRECOND_RUIZ_HPP
#define PROXSUITE_QP_DENSE_PRECOND_RUIZ_HPP

#include "qp/dense/views.hpp"
#include "qp/dense/fwd.hpp"
#include <linearsolver/dense/core.hpp>
#include <ostream>

#include <Eigen/Core>

#include <iostream>

namespace proxsuite {
namespace qp {
enum struct Symmetry {
	general,
	lower,
	upper,
};
namespace dense {
namespace detail {

template <typename T>
auto ruiz_scale_qp_in_place( //
		VectorViewMut<T> delta_,
		VectorViewMut<T> tmp_delta_preallocated,
		std::ostream* logger_ptr,
		QpViewBoxMut<T> qp,
		T epsilon,
		isize max_iter,
		Symmetry sym) -> T {

	T c(1);
	auto S = delta_.to_eigen();

	auto H = qp.H.to_eigen();
	auto g = qp.g.to_eigen();
	auto A = qp.A.to_eigen();
	auto b = qp.b.to_eigen();
	auto C = qp.C.to_eigen();
	auto u = qp.u.to_eigen();
	auto l = qp.l.to_eigen();

	static constexpr T machine_eps = std::numeric_limits<T>::epsilon();
	/*
	 * compute equilibration parameters and scale in place the qp following
	 * algorithm
	 *
	 * modified: removed g in gamma computation
	 */

	isize n = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;

	T gamma = T(1);

	auto delta = tmp_delta_preallocated.to_eigen();

	i64 iter = 1;

	while (infty_norm((1 - delta.array()).matrix()) > epsilon) {
		if (logger_ptr != nullptr) {
			*logger_ptr                                                //
					<< "j : "                                              //
					<< iter                                                //
					<< " ; error : "                                       //
					<< infty_norm((1 - delta.array()).matrix()) //
					<< "\n\n";
		}
		if (iter == max_iter) {
			break;
		} else {
			++iter;
		}

		// normalization vector
		{
			for (isize k = 0; k < n; ++k) {
				switch (sym) {
				case Symmetry::upper: { // upper triangular part
					delta(k) = T(1) / (sqrt(std::max({
																 infty_norm(H.col(k).head(k)),
																 infty_norm(H.row(k).tail(n - k)),
																 infty_norm(A.col(k)),
																 infty_norm(C.col(k)),
														 })) +
					                   machine_eps);
					break;
				}
				case Symmetry::lower: { // lower triangular part
					delta(k) = T(1) / (sqrt(std::max({
																 infty_norm(H.row(k).head(k)),
																 infty_norm(H.col(k).tail(n - k)),
																 infty_norm(A.col(k)),
																 infty_norm(C.col(k)),
														 })) +
					                   machine_eps);
					break;
				}
				case Symmetry::general: {
					delta(k) = T(1) / (sqrt(std::max({
																 infty_norm(H.col(k)),
																 infty_norm(A.col(k)),
																 infty_norm(C.col(k)),
														 })) +
					                   machine_eps);

					break;
				}
				}
			}

			for (isize k = 0; k < n_eq; ++k) {
				T aux = sqrt(infty_norm(A.row(k)));
				delta(n + k) = T(1) / (aux + machine_eps);
			}
			for (isize k = 0; k < n_in; ++k) {
				T aux = sqrt(infty_norm(C.row(k)));
				delta(k + n + n_eq) = T(1) / (aux + machine_eps);
			}
		}
		{

			// normalize A and C
			A = delta.segment(n, n_eq).asDiagonal() * A * delta.head(n).asDiagonal();
			C = delta.tail(n_in).asDiagonal() * C * delta.head(n).asDiagonal();
			// normalize vectors
			g.array() *= delta.head(n).array();
			b.array() *= delta.middleRows(n, n_eq).array();
			u.array() *= delta.tail(n_in).array();
			l.array() *= delta.tail(n_in).array();

			// normalize H
			switch (sym) {
			case Symmetry::upper: {
				// upper triangular part
				for (isize j = 0; j < n; ++j) {
					H.col(j).head(j + 1) *= delta(j);
				}
				// normalisation des lignes
				for (isize i = 0; i < n; ++i) {
					H.row(i).tail(n - i) *= delta(i);
				}
				break;
			}
			case Symmetry::lower: {
				// lower triangular part
				for (isize j = 0; j < n; ++j) {
					H.col(j).tail(n - j) *= delta(j);
				}
				// normalisation des lignes
				for (isize i = 0; i < n; ++i) {
					H.row(i).head(i + 1) *= delta(i);
				}
				break;
			}
			case Symmetry::general: {
				// all matrix
				H = delta.head(n).asDiagonal() * H * delta.head(n).asDiagonal();
				break;
			}
			default:
				break;
			}

			// additional normalization for the cost function
			switch (sym) {
			case Symmetry::upper: {
				// upper triangular part
				T tmp = T(0);
				for (isize j = 0; j < n; ++j) {
					tmp += qp::dense::infty_norm(H.row(j).tail(n - j));
				}
				gamma = 1 / std::max(tmp / T(n), T(1));
				break;
			}
			case Symmetry::lower: {
				// lower triangular part
				T tmp = T(0);
				for (isize j = 0; j < n; ++j) {
					tmp += qp::dense::infty_norm(H.col(j).tail(n - j));
				}
				gamma = 1 / std::max(tmp / T(n), T(1));
				break;
			}
			case Symmetry::general: {
				// all matrix
				gamma =
						1 /
						std::max(
								T(1), (H.colwise().template lpNorm<Eigen::Infinity>()).mean());
				break;
			}
			default:
				break;
			}

			g *= gamma;
			H *= gamma;

			S.array() *= delta.array(); // coefficientwise product
			c *= gamma;
		}
	}
	return c;
}
} // namespace detail

namespace preconditioner {

template <typename T>
struct RuizEquilibration {
	Vec<T> delta;
	T c;
	isize dim;
	T epsilon;
	i64 max_iter;
	Symmetry sym;

	std::ostream* logger_ptr = nullptr;

	explicit RuizEquilibration(
			isize dim_,
			isize n_eq_in,
			T epsilon_ = T(1e-3),
			i64 max_iter_ = 10,
			Symmetry sym_ = Symmetry::general,
			std::ostream* logger = nullptr)
			: delta(dim_ + n_eq_in),
				c(1),
				dim(dim_),
				epsilon(epsilon_),
				max_iter(max_iter_),
				sym(sym_),
				logger_ptr(logger) {
		delta.setOnes();
	}
	void print() {
		// CHANGE: endl to newline
		*logger_ptr << " delta : " << delta << "\n\n";
		*logger_ptr << " c : " << c << "\n\n";
	}

	static auto
	scale_qp_in_place_req(veg::Tag<T> tag, isize n, isize n_eq, isize n_in)
			-> veg::dynstack::StackReq {
		return linearsolver::dense::temp_vec_req(tag, n + n_eq + n_in);
	}

	// H_new = c * head @ H @ head
	// A_new = tail @ A @ head
	// g_new = c * head @ g
	// b_new = tail @ b
	void scale_qp_in_place(QpViewBoxMut<T> qp, veg::dynstack::DynStackMut stack) {
		delta.setOnes();
		LDLT_TEMP_VEC(T, tmp_delta, qp.H.rows + qp.A.rows + qp.C.rows, stack);
		c = detail::ruiz_scale_qp_in_place(
				{qp::from_eigen, delta},
				{qp::from_eigen, tmp_delta},
				logger_ptr,
				qp,
				epsilon,
				max_iter,
				sym);
	}
	void scale_qp(
			QpViewBox<T> qp,
			QpViewBoxMut<T> scaled_qp,
			VectorViewMut<T> tmp_delta_preallocated) {

		/*
		 * scaled_qp is scaled, whereas first qp is not
		 * the procedure computes as well equilibration parameters using default
		 * parameters
		 */

		scaled_qp.H.to_eigen() = qp.H.to_eigen();
		scaled_qp.A.to_eigen() = qp.A.to_eigen();
		scaled_qp.C.to_eigen() = qp.C.to_eigen();
		scaled_qp.g.to_eigen() = qp.g.to_eigen();
		scaled_qp.b.to_eigen() = qp.b.to_eigen();
		scaled_qp.d.to_eigen() = qp.d.to_eigen();

		scale_qp_in_place(scaled_qp, tmp_delta_preallocated, epsilon, max_iter);
	}
	// modifies variables in place
	void scale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.array().head(dim);
	}
	void scale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() /
		                          delta.tail(delta.size() - dim).array() * c;
	}

	void scale_dual_in_place_eq(VectorViewMut<T> dual) {
		dual.to_eigen().array() =
				dual.as_const().to_eigen().array() /
				delta.middleRows(dim, dual.to_eigen().size()).array() * c;
	}
	void scale_dual_in_place_in(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() /
		                          delta.tail(dual.to_eigen().size()).array() * c;
	}

	void unscale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.array().head(dim);
	}
	void unscale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() *
		                          delta.tail(delta.size() - dim).array() / c;
	}

	void unscale_dual_in_place_eq(VectorViewMut<T> dual) {
		dual.to_eigen().array() =
				dual.as_const().to_eigen().array() *
				delta.middleRows(dim, dual.to_eigen().size()).array() / c;
	}

	void unscale_dual_in_place_in(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() *
		                          delta.tail(dual.to_eigen().size()).array() / c;
	}
	// modifies residuals in place
	void scale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.tail(delta.size() - dim).array();
	}

	void scale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) {
		primal_eq.to_eigen().array() *=
				delta.middleRows(dim, primal_eq.to_eigen().size()).array();
	}
	void scale_primal_residual_in_place_in(VectorViewMut<T> primal_in) {
		primal_in.to_eigen().array() *=
				delta.tail(primal_in.to_eigen().size()).array();
	}
	void scale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() *= delta.head(dim).array() * c;
	}
	void unscale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.tail(delta.size() - dim).array();
	}
	void unscale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) {
		primal_eq.to_eigen().array() /=
				delta.middleRows(dim, primal_eq.to_eigen().size()).array();
	}
	void unscale_primal_residual_in_place_in(VectorViewMut<T> primal_in) {
		primal_in.to_eigen().array() /=
				delta.tail(primal_in.to_eigen().size()).array();
	}
	void unscale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() /= delta.head(dim).array() * c;
	}
};

} // namespace preconditioner
} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_PRECOND_RUIZ_HPP */
