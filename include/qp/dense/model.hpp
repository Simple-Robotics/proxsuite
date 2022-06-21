/** \file */
#ifndef PROXSUITE_QP_DENSE_MODEL_HPP
#define PROXSUITE_QP_DENSE_MODEL_HPP

#include <Eigen/Core>
#include <veg/type_traits/core.hpp>
#include "qp/dense/fwd.hpp"

namespace proxsuite {
namespace qp {
namespace dense {
///
/// @brief This class stores the model of the QP problem.
///
/*!
 * Model class of the dense solver storing the QP problem.
*/
template <typename T>
struct Model {

	///// QP STORAGE
	Mat<T> H;
	Vec<T> g;
	Mat<T> A;
	Mat<T> C;
	Vec<T> b;
	Vec<T> u;
	Vec<T> l;

	///// model size
	isize dim;
	isize n_eq;
	isize n_in;
	isize n_total;
	/*!
	 * Default constructor.
	 * @param _dim primal variable dimension.
	 * @param _n_eq number of equality constraints.
	 * @param _n_in number of inequality constraints.
	 */
	Model(isize _dim, isize _n_eq, isize _n_in)
			: H(_dim, _dim),
				g(_dim),
				A(_n_eq, _dim),
				C(_n_in, _dim),
				b(_n_eq),
				u(_n_in),
				l(_n_in),
				dim(_dim),
				n_eq(_n_eq),
				n_in(_n_in),
				n_total(_dim + _n_eq + _n_in) {

		H.setZero();
		g.setZero();
		A.setZero();
		C.setZero();
		b.setZero();
		u.setZero();
		l.setZero();
	}
};

} // namespace dense
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_MODEL_HPP */
