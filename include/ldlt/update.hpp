#ifndef INRIA_LDLT_UPDATE_HPP_OHWFTYRXS
#define INRIA_LDLT_UPDATE_HPP_OHWFTYRXS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {

template <typename Scalar, Layout L>
void rank1_update(
		MatrixViewMut<Scalar, L> l,
		VectorViewMut<Scalar> d,
		VectorView<Scalar> z,
		Scalar alpha) {

	i32 dim = l.dim;
	auto _workspace = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>{dim};
	auto w = VecMapMut<Scalar>{_workspace.data(), dim};
	w = VecMap<Scalar>{z.data, dim};

	for (i32 j = 0; j < dim; ++j) {
		Scalar p = w(j);
		Scalar dj = d(j);
		Scalar new_dj = (dj + alpha * p * p);
		Scalar gamma = dj / new_dj;
		d(j) = new_dj;
		Scalar beta = p * alpha / new_dj;
		alpha *= gamma;

		Scalar c = (gamma + beta * p);
		for (i32 r = j + 1; r < dim; ++r) {
			w(r) -= p * l(r, j);
			l(r, j) = c * l(r, j) + beta * w(r);
		}
	}
}
} // namespace detail

namespace nb {
struct rank1_update {
	template <typename Scalar, Layout L>
	void operator()(
			MatrixViewMut<Scalar, L> l,
			VectorViewMut<Scalar> d,
			VectorView<Scalar> z,
			Scalar alpha) {
		detail::rank1_update(l, d, z, alpha);
	}
};
} // namespace nb
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_UPDATE_HPP_OHWFTYRXS */
