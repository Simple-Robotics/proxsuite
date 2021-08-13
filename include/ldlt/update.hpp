#ifndef INRIA_LDLT_UPDATE_HPP_OHWFTYRXS
#define INRIA_LDLT_UPDATE_HPP_OHWFTYRXS

#include "ldlt/views.hpp"

namespace ldlt {
namespace detail {

template <typename Scalar, Layout L>
LDLT_NO_INLINE void rank1_update(
		LdltViewMut<Scalar, L> out,
		LdltView<Scalar, L> in,
		VectorViewMut<Scalar> z,
		i32 offset,
		Scalar alpha) {

	i32 dim = out.l.rows;
	Scalar* HEDLEY_RESTRICT wp = z.data;

	for (i32 j = offset; j < dim; ++j) {
		Scalar p = wp[j - offset];
		Scalar old_dj = in.d(j);
		Scalar new_dj = old_dj + alpha * p * p;
		Scalar gamma = old_dj / new_dj;
		out.d(j) = new_dj;
		Scalar beta = p * alpha / new_dj;
		alpha *= gamma;

		Scalar c = gamma + beta * p;
		for (i32 r = j + 1; r < dim; ++r) {
			LDLT_FP_PRAGMA
			auto& wr = wp[r - offset];
			wr = wr - p * in.l(r, j);
			out.l(r, j) = c * in.l(r, j) + beta * wr;
		}
	}
}

template <typename Scalar, Layout L>
LDLT_NO_INLINE void mu_update_one_pass(
		LdltViewMut<Scalar, L> out,
		LdltView<Scalar, L> in,
		VectorView<Scalar> diag_diff,
		i32 start_index) {

	i32 dim = out.l.rows;
	i32 m = diag_diff.dim;
	if (m == 0) {
		return;
	}
	i32 dim_rest = dim - start_index;

	// TODO: fuse allocations, split up to different cache lines
	LDLT_WORKSPACE_MEMORY(ws, m * dim_rest, Scalar);
	LDLT_WORKSPACE_MEMORY(alphas, m, Scalar);
	LDLT_WORKSPACE_MEMORY(ps, m, Scalar);
	LDLT_WORKSPACE_MEMORY(betas, m, Scalar);
	LDLT_WORKSPACE_MEMORY(cs, m, Scalar);

	for (i32 k = 0; k < m * dim_rest; ++k) {
		ws[k] = Scalar(0);
	}
	for (i32 k = 0; k < m; ++k) {
		ws[k * dim_rest + k] = Scalar(1);
		alphas[k] = diag_diff(k);
	}

	for (i32 j = start_index; j < dim; ++j) {

		{
			Scalar dj = in.d(j);
			for (i32 k = 0, w_offset = 0; //
			     k < m;
			     ++k, w_offset += dim_rest) {
				Scalar p = ws[w_offset + (j - start_index)];
				Scalar& alpha = alphas[k];

				Scalar old_dj = dj;
				Scalar new_dj = old_dj + alpha * p * p;
				Scalar gamma = old_dj / new_dj;
				Scalar beta = p * alpha / new_dj;
				Scalar c = gamma + beta * p;

				alpha *= gamma;

				dj = new_dj;
				ps[k] = p;
				betas[k] = beta;
				cs[k] = c;
			}
			out.d(j) = dj;
		}

		// TODO: vectorize
		for (i32 r = j + 1; r < dim; ++r) {
			i32 max_k = std::min(r - j, m);

			Scalar lr = in.l(r, j);
			for (i32 k = 0, w_offset = 0; //
			     k < max_k;
			     ++k, w_offset += dim_rest) {
				LDLT_FP_PRAGMA
				auto& wr = ws[w_offset + (r - start_index)];
				wr = wr - ps[k] * lr;
				lr = cs[k] * lr + betas[k] * wr;
			}
			out.l(r, j) = lr;
		}
	}
}

} // namespace detail

extern template void ldlt::detail::rank1_update(
		LdltViewMut<f32, colmajor>,
		LdltView<f32, colmajor>,
		VectorViewMut<f32>,
		i32,
		f32);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f32, rowmajor>,
		LdltView<f32, rowmajor>,
		VectorViewMut<f32>,
		i32,
		f32);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f64, colmajor>,
		LdltView<f64, colmajor>,
		VectorViewMut<f64>,
		i32,
		f64);
extern template void ldlt::detail::rank1_update(
		LdltViewMut<f64, rowmajor>,
		LdltView<f64, rowmajor>,
		VectorViewMut<f64>,
		i32,
		f64);

namespace nb {
struct rank1_update {
	template <typename Scalar, Layout L>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, L> out,
			LdltView<Scalar, L> in,
			VectorView<Scalar> z,
			Scalar alpha) const {
		i32 dim = out.l.rows;
		LDLT_WORKSPACE_MEMORY(wp, out.d.dim, Scalar);
		std::copy(z.data, z.data + dim, wp);
		detail::rank1_update(out, in, VectorViewMut<Scalar>{wp, dim}, 0, alpha);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(rank1_update);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_UPDATE_HPP_OHWFTYRXS */
