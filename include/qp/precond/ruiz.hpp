#ifndef INRIA_LDLT_RUIZ_HPP_XXCS7AMRS
#define INRIA_LDLT_RUIZ_HPP_XXCS7AMRS

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include "ldlt/views.hpp"
#include "ldlt/detail/meta.hpp"
#include "qp/views.hpp"
#include <ostream>

#include <Eigen/Core>

namespace qp {
enum struct Symmetry {
	general,
	lower,
	upper,
};

namespace detail {
namespace nb {
struct sqrt {
	template <typename T>
	auto operator()(T x) const -> T {
		using std::sqrt;
		return sqrt(x);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(sqrt);

template <typename T>
auto ruiz_scale_qp_in_place( //
		VectorViewMut<T> delta_,
		std::ostream* logger_ptr,
		qp::QpViewMut<T> qp,
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
	auto d = qp.d.to_eigen();

	T machine_eps = std::numeric_limits<T>::epsilon();
	/*
	 * compute equilibration parameters and scale in place the qp following
	 * algorithm 1 at
	 * https://sharelatex.irisa.fr/project/6100343e095d2649760f5631
	 */

	isize n = qp.H.rows;
	isize n_eq = qp.A.rows;
	isize n_in = qp.C.rows;

	S.setConstant(T(1));
	T gamma = T(1);

	LDLT_WORKSPACE_MEMORY(_delta, Vec(n + n_eq + n_in), T);
	auto delta = _delta.to_eigen();

	delta.setZero();

	i64 iter = 1;
	while (infty_norm((1 - delta.array()).matrix()) > epsilon) {
		if (logger_ptr != nullptr) {
			*logger_ptr                                     //
					<< "j : "                                   //
					<< iter                                     //
					<< " ; error : "                            //
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
			LDLT_DECL_SCOPE_TIMER("ruiz equilibration", "delta computation", T);
			for (isize k = 0; k < n; ++k) {
				switch (sym) {
				case Symmetry::upper: { // upper triangular part
					delta(k) = T(1) / (sqrt(max2(
																 infty_norm(H.row(k).tail(n - k)), //
																 max2(                             //
																		 infty_norm(A.col(k)),
																		 infty_norm(C.col(k))))) +
					                   machine_eps);
					break;
				}
				case Symmetry::lower: { // lower triangular part
					delta(k) = T(1) / (sqrt(max2(
																 infty_norm(H.col(k).tail(n - k)), //
																 max2(                             //
																		 infty_norm(A.col(k)),
																		 infty_norm(C.col(k))))) +
					                   machine_eps);
					break;
				}
				case Symmetry::general: {
					delta(k) = T(1) / (sqrt(max2(
																 infty_norm(H.col(k)), //
																 max2(                 //
																		 infty_norm(A.col(k)),
																		 infty_norm(C.col(k))))) +
					                   machine_eps);
					break;
				}
				default: {
				}
				}
			}

			for (isize k = 0; k < n_eq; ++k) {
				delta(n + k) = T(1) / (sqrt(infty_norm(A.row(k))) + machine_eps);
			}
			for (isize k = 0; k < n_in; ++k) {
				delta(k + n + n_eq) = T(1) / (sqrt(infty_norm(C.row(k))) + machine_eps);
			}
		}

		{
			LDLT_DECL_SCOPE_TIMER("ruiz equilibration", "normalization", T);

			// normalize A and C
			A = delta.middleRows(n, n_eq).asDiagonal() * A *
			    delta.head(n).asDiagonal();
			C = delta.tail(n_in).asDiagonal() * C * delta.head(n).asDiagonal();
			// normalize vectors
			g.array() *= delta.head(n).array();
			b.array() *= delta.middleRows(n, n_eq).array();
			d.array() *= delta.tail(n_in).array();

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
					tmp += infty_norm(H.row(j).tail(n - j));
				}
				gamma = 1 / max2(tmp / T(n), max2(infty_norm(g), T(1)));
				break;
			}
			case Symmetry::lower: {
				// lower triangular part
				T tmp = T(0);
				for (isize j = 0; j < n; ++j) {
					tmp += infty_norm(H.col(j).tail(n - j));
				}
				gamma = 1 / max2(tmp / T(n), max2(infty_norm(g), T(1)));
				break;
			}
			case Symmetry::general: {
				// all matrix
				gamma =
						1 / max2(
										max2(infty_norm(g), T(1)),
										(H.colwise().template lpNorm<Eigen::Infinity>()).mean());
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
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

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
			i64 max_iter_ = 20,
			Symmetry sym_ = Symmetry::general,
			std::ostream* logger = nullptr)
			: delta(dim_ + n_eq_in),
				c(1),
				dim(dim_),
				epsilon(epsilon_),
				max_iter(max_iter_),
				sym(sym_),
				logger_ptr(logger) {
		delta.setZero();
	}
	void print() {
		// CHANGE: endl to newline
		*logger_ptr << " delta : " << delta << "\n\n";
		*logger_ptr << " c : " << c << "\n\n";
	}

	// H_new = c * head @ H @ head
	// A_new = tail @ A @ head
	// g_new = c * head @ g
	// b_new = tail @ b
	void scale_qp_in_place(QpViewMut<T> qp) {
		c = detail::ruiz_scale_qp_in_place(
				{ldlt::from_eigen, delta}, logger_ptr, qp, epsilon, max_iter, sym);
	}
	void scale_qp(QpView<T> qp, QpViewMut<T> scaled_qp) {

		using namespace detail;
		/*
		 * scaled_qp is scaled, whereas first qp is not
		 * the procedure computes as well equilibration parameters using default
		 * parameters following algorithm 1 at
		 * https://sharelatex.irisa.fr/project/6100343e095d2649760f5631
		 */

		scaled_qp.H.to_eigen() = qp.H.to_eigen();
		scaled_qp.A.to_eigen() = qp.A.to_eigen();
		scaled_qp.C.to_eigen() = qp.C.to_eigen();
		scaled_qp.g.to_eigen() = qp.g.to_eigen();
		scaled_qp.b.to_eigen() = qp.b.to_eigen();
		scaled_qp.d.to_eigen() = qp.d.to_eigen();

		scale_qp_in_place(scaled_qp, epsilon, max_iter);
	}
	// modifies variables in place
	void scale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.array().head(dim);
	}
	void scale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() /
		                          delta.tail(delta.size() - dim).array() * c;
	}
	void unscale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.array().head(dim);
	}
	void unscale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() *
		                          delta.tail(delta.size() - dim).array() / c;
	}

	// modifies residuals in place
	void scale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.tail(delta.size() - dim).array();
	}
	void scale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() *= delta.head(dim).array() * c;
	}
	void unscale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.tail(delta.size() - dim).array();
	}
	void unscale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() /= delta.head(dim).array() * c;
	}
};

} // namespace preconditioner

} // namespace qp

#endif /* end of include guard INRIA_LDLT_RUIZ_HPP_XXCS7AMRS */
