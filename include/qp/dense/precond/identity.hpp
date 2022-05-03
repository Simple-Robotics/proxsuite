#ifndef PROXSUITE_INCLUDE_QP_DENSE_PRECOND_IDENTITY_HPP
#define PROXSUITE_INCLUDE_QP_DENSE_PRECOND_IDENTITY_HPP

#include "qp/dense/dense-views.hpp"

namespace qp {
namespace dense{
namespace preconditioner {
struct IdentityPrecond {
	template <typename T>
	void scale_qp_in_place(QpViewBoxMut<T> /*qp*/) const noexcept {}

	template <typename T>
	void scale_primal_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void scale_dual_in_place_in(VectorViewMut<T> /*y*/) const noexcept {}
	template <typename T>
	void scale_dual_in_place_eq(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void scale_primal_residual_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void scale_dual_residual_in_place(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void unscale_primal_in_place(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void unscale_dual_in_place_in(VectorViewMut<T> /*y*/) const noexcept {}
	template <typename T>
	void unscale_dual_in_place_eq(VectorViewMut<T> /*y*/) const noexcept {}

	template <typename T>
	void
	unscale_primal_residual_in_place_in(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void
	unscale_primal_residual_in_place_eq(VectorViewMut<T> /*x*/) const noexcept {}
	template <typename T>
	void unscale_dual_residual_in_place(VectorViewMut<T> /*y*/) const noexcept {}
};
} // namespace preconditioner
} // namespace dense
} // namespace qp

#endif /* end of include guard PROXSUITE_INCLUDE_QP_DENSE_PRECOND_IDENTITY_HPP */
