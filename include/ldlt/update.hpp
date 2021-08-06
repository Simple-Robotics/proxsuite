#ifndef INRIA_LDLT_UPDATE_HPP_OHWFTYRXS
#define INRIA_LDLT_UPDATE_HPP_OHWFTYRXS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {

template <typename Scalar, Layout L>
LDLT_NO_INLINE void rank1_update(
		LdltViewMut<Scalar, L> out,
		LdltView<Scalar, L> in,
		VectorView<Scalar> z,
		Scalar alpha) {

	i32 dim = out.l.dim;
	auto _workspace = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>{dim};
	auto w = VecMapMut<Scalar>{_workspace.data(), dim};
	w = VecMap<Scalar>{z.data, dim};

	Scalar* HEDLEY_RESTRICT wp = w.data();

	for (i32 j = 0; j < dim; ++j) {
		Scalar p = wp[j];
		Scalar dj = in.d(j);
		Scalar new_dj = (dj + alpha * p * p);
		Scalar gamma = dj / new_dj;
		out.d(j) = new_dj;
		Scalar beta = p * alpha / new_dj;
		alpha *= gamma;

		Scalar c = (gamma + beta * p);
		for (i32 r = j + 1; r < dim; ++r) {
			LDLT_FP_PRAGMA
			wp[r] -= p * in.l(r, j);
			out.l(r, j) = c * in.l(r, j) + beta * wp[r];
		}
	}
}
} // namespace detail

extern template void ldlt::detail::rank1_update(
		LdltViewMut<f32, colmajor>, LdltView<f32, colmajor>, VectorView<f32>, f32);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f32, rowmajor>, LdltView<f32, rowmajor>, VectorView<f32>, f32);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f64, colmajor>, LdltView<f64, colmajor>, VectorView<f64>, f64);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f64, rowmajor>, LdltView<f64, rowmajor>, VectorView<f64>, f64);

namespace nb {
struct rank1_update {
	template <typename Scalar, Layout L>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, L> out,
			LdltView<Scalar, L> in,
			VectorView<Scalar> z,
			Scalar alpha) const {
		detail::rank1_update(out, in, z, alpha);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(rank1_update);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_UPDATE_HPP_OHWFTYRXS */
