#ifndef INRIA_LDLT_RUIZ_HPP_XXCS7AMRS
#define INRIA_LDLT_RUIZ_HPP_XXCS7AMRS

#include "ldlt/detail/tags.hpp"
#include "ldlt/detail/macros.hpp"
#include "ldlt/detail/simd.hpp"
#include "ldlt/views.hpp"
#include "ldlt/qp_eq.hpp"
#include <ostream>

#include <Eigen/Core>

namespace ldlt {
namespace detail {
namespace nb {
struct sqrt {
	template <typename Scalar>
	auto operator()(Scalar x) const -> Scalar {
		using std::sqrt;
		return sqrt(x);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(sqrt);
} // namespace detail
namespace qp {
namespace preconditioner {

template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename Scalar, Layout LH, Layout LC>
struct RuizEquilibration {
	Vec<Scalar> delta;
	Scalar c;
	i32 dim;
	std::ostream* logger_ptr = nullptr;

	explicit RuizEquilibration(
			i32 dim, i32 n_eq_in, std::ostream* logger = nullptr)
			: delta(dim + n_eq_in), c(1), dim(dim), logger_ptr(logger) {
		delta.setZero();
	};
	void print() {
		// CHANGE: endl to newline
		*logger_ptr << " delta : " << delta << "\n\n";
		*logger_ptr << " c : " << c << "\n\n";
	};

	// H_new = c * head @ H @ head
	// A_new = tail @ A @ head
	// g_new = c * head @ g
  // b_new = tail @ b
	void scale_qp_in_place(
			QpViewMut<Scalar, LH, LC> qp,
			Scalar epsilon = Scalar(1.e-3),
			i32 max_iter = 20) {
		using detail::max2;
		using detail::sqrt;
		using detail::norm_inf;
		using detail::to_eigen_matrix_mut;
		using detail::to_eigen_matrix;
		using detail::to_eigen_vector_mut;
		using detail::to_eigen_vector;
		Scalar machine_eps = std::numeric_limits<Scalar>::epsilon();
		/*
		 * compute equilibration parameters and scale in place the qp following
		 * algorithm 1 at
		 * https://sharelatex.irisa.fr/project/6100343e095d2649760f5631
		 */

		i32 n = qp.H.rows;
		i32 n_eq = qp.A.rows;
		i32 n_in = qp.C.rows;
		const i32 tot_dim = n + n_eq + n_in;

		i32 iter = 1;
		while (norm_inf((1 - delta.array()).matrix()) > epsilon) {
			if (iter >= max_iter) {
				if (logger_ptr != nullptr) {
					*logger_ptr                                   //
							<< "j : "                                 //
							<< iter                                   //
							<< " ; error : "                          //
							<< norm_inf((1 - delta.array()).matrix()) //
							<< "\n\n";
				}
				break;
			} else {
				if (logger_ptr != nullptr) {
					*logger_ptr                                   //
							<< "j : "                                 //
							<< iter                                   //
							<< " ; error : "                          //
							<< norm_inf((1 - delta.array()).matrix()) //
							<< "\n\n";
				}
				iter += 1;
			}
			// vecteur de normalisation
			for (i32 k = 0; k < tot_dim; ++k) {
				if (k < n) {
					// convertir en matrice, utiliser eigen, puis repasser en view

					delta(k) =
							Scalar(1) / (sqrt(max2(
															 max2(
																	 norm_inf(to_eigen_matrix_mut(qp.H).col(k)),
																	 norm_inf(to_eigen_matrix_mut(qp.A).col(k))),
															 norm_inf(to_eigen_matrix_mut(qp.C).col(k)))) +
					                 machine_eps); // to avoid
					                               // concatenation
					                               // throught temporary
					                               // memory allocation
				}

				else if (n <= k and k < n + n_eq) {
					delta(k) = Scalar(1) /
					           (sqrt(norm_inf((to_eigen_matrix_mut(qp.A)).row(k - n))) +
					            machine_eps);
				} else // k >= n+n_eq
				{
					delta(k) =
							Scalar(1) /
							(sqrt(norm_inf((to_eigen_matrix_mut(qp.C)).row(k - n - n_eq))) +
					     machine_eps);
				}
			}

			// normalisation des colonnes de H, A, C

			for (i32 j = 0; j < n; ++j) {

				to_eigen_matrix_mut(qp.H).col(j) *= delta(j);
				to_eigen_matrix_mut(qp.A).col(j) *= delta(j);
				to_eigen_matrix_mut(qp.C).col(j) *= delta(j);
			}
			// normalisation des lignes
			for (i32 i = 0; i < n; ++i) {
				to_eigen_matrix_mut(qp.H).row(i) *= delta(i);
				to_eigen_vector_mut(qp.g)(i) *= delta(i);
			}

			for (i32 i = 0; i < n_eq; ++i) {
				to_eigen_matrix_mut(qp.A).row(i) *= delta(i + n);
				qp.b(i) *= delta(i + n);
			}

			for (i32 i = 0; i < n_in; ++i) {
				to_eigen_matrix_mut(qp.C).row(i) *= delta(i + n + n_eq);
				to_eigen_vector_mut(qp.d)(i) *= delta(i + n + n_eq);
			}

			// supplementary normalisation for the cost function

			Scalar gamma =
					1 / max2(
									max2(norm_inf(to_eigen_vector_mut(qp.g)), Scalar(1)),
									(to_eigen_matrix_mut(qp.H)
			                 .colwise()
			                 .template lpNorm<Eigen::Infinity>())
											.mean());

			to_eigen_vector_mut(qp.g) *= gamma;
			to_eigen_matrix_mut(qp.H) *= gamma;

			delta.array() *= delta.array(); // coefficient wise product using array
			c *= gamma;
		}
	}
	void scale_qp(
			QpView<Scalar, LH, LC> qp,
			QpViewMut<Scalar, LH, LC> scaled_qp,
			Scalar epsilon = Scalar(1.e-3F),
			i32 max_iter = 20) {

		using detail::to_eigen_matrix_mut;
		using detail::to_eigen_matrix;
		using detail::to_eigen_vector_mut;
		using detail::to_eigen_vector;
		/*
		 * scaled_qp is scaled, whereas first qp is not
		 * the procedure computes as well equilibration parameters using default
		 * parameters following algorithm 1 at
		 * https://sharelatex.irisa.fr/project/6100343e095d2649760f5631
		 */

		to_eigen_matrix_mut(scaled_qp.H) = to_eigen_matrix(qp.H);
		to_eigen_matrix_mut(scaled_qp.A) = to_eigen_matrix(qp.A);
		to_eigen_matrix_mut(scaled_qp.C) = to_eigen_matrix(qp.C);
		to_eigen_vector_mut(scaled_qp.g) = to_eigen_vector(qp.g);
		to_eigen_vector_mut(scaled_qp.b) = to_eigen_vector(qp.b);
		to_eigen_vector_mut(scaled_qp.d) = to_eigen_vector(qp.d);

		scale_qp_in_place(scaled_qp, epsilon, max_iter);
	}
	// modifies variables in place
	void scale_primal_in_place(VectorViewMut<Scalar> primal) {
		detail::to_eigen_vector_mut(primal).array() /= delta.array().head(dim);
	}
	void scale_dual_in_place(VectorViewMut<Scalar> dual) {
		detail::to_eigen_vector_mut(dual).array() =
				detail::to_eigen_vector(dual.as_const()).array() /
				delta.tail(delta.size() - dim).array() * c;
	}
	void unscale_primal_in_place(VectorViewMut<Scalar> primal) {
		detail::to_eigen_vector_mut(primal).array() *= delta.array().head(dim);
	}
	void unscale_dual_in_place(VectorViewMut<Scalar> dual) {
		detail::to_eigen_vector_mut(dual).array() =
				detail::to_eigen_vector(dual.as_const()).array() *
				delta.tail(delta.size() - dim).array() / c;
	}

	// modifies residuals in place
	void scale_primal_residue_in_place(VectorViewMut<Scalar> primal) {
		detail::to_eigen_vector_mut(primal).array() *=
				delta.tail(delta.size() - dim).array();
	}
	void scale_dual_residue_in_place(VectorViewMut<Scalar> dual) {
		detail::to_eigen_vector_mut(dual).array() *= delta.head(dim).array() * c;
	}
	void unscale_primal_residue_in_place(VectorViewMut<Scalar> primal) {
		detail::to_eigen_vector_mut(primal).array() /=
				delta.tail(delta.size() - dim).array();
	}
	void unscale_dual_residue_in_place(VectorViewMut<Scalar> dual) {
		detail::to_eigen_vector_mut(dual).array() /= delta.head(dim).array() * c;
	}
};

} // namespace preconditioner
} // namespace qp
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_RUIZ_HPP_XXCS7AMRS */
