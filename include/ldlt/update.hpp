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

	Scalar acc(1);

	for (i32 j = 0; j < dim; ++j) {
		Scalar wj = w(j);
		Scalar dj = d(j);

		Scalar a = alpha * wj * wj;
		Scalar b = dj * acc + a;

		d(j) += a / acc;
		acc += a / dj;

		for (i32 r = j + 1; r < dim; ++r) {
			w(r) -= wj * l(r, j);
			l(r, j) += (alpha * wj / b) * w(r);
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
