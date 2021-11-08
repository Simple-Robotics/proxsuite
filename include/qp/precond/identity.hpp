#ifndef INRIA_LDLT_IDENTITY_HPP_V2BH4W5SS
#define INRIA_LDLT_IDENTITY_HPP_V2BH4W5SS

#include "qp/views.hpp"

namespace qp {
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
} // namespace qp

#endif /* end of include guard INRIA_LDLT_IDENTITY_HPP_V2BH4W5SS */
