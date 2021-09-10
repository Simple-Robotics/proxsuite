#ifndef INRIA_LDLT_UPDATE_HPP_OHWFTYRXS
#define INRIA_LDLT_UPDATE_HPP_OHWFTYRXS

#include "ldlt/views.hpp"
#include "ldlt/detail/meta.hpp"

namespace ldlt {

namespace diagonal_update_strategies {
LDLT_DEFINE_TAG(multi_pass, MultiPass);
LDLT_DEFINE_TAG(single_pass, SinglePass);
LDLT_DEFINE_TAG(refactorize, Refactorize);
LDLT_DEFINE_TAG(full_refactorize, FullRefactorize);
} // namespace diagonal_update_strategies

namespace detail {

template <typename T, usize N>
LDLT_INLINE void rank1_update_inner_loop_packed(
		isize i,
		Pack<T, N> p_p,
		Pack<T, N> p_c,
		Pack<T, N> p_beta,

		MatrixViewMut<T, colmajor> out_l,
		MatrixView<T, colmajor> in_l,
		T* wp,
		isize j,
		isize offset) {
	isize r = i + j + 1;

	// TODO[PERF]: check asm, clang does weird stuff with address computations in
	// tight loop

	auto in_l_ptr = in_l.ptr(r, j);
	auto out_l_ptr = out_l.ptr(r, j);

	Pack<T, N> p_wr = Pack<T, N>::load_unaligned(wp + (r - offset));
	Pack<T, N> p_in_l = Pack<T, N>::load_unaligned(in_l_ptr);

	p_wr = Pack<T, N>::fnmadd(p_p, p_in_l, p_wr);
	p_wr.store_unaligned(wp + (r - offset));
	p_in_l = p_in_l.mul(p_c);
	p_in_l = Pack<T, N>::fmadd(p_beta, p_wr, p_in_l);

	p_in_l.store_unaligned(out_l_ptr);
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, true> /*tag*/,
		MatrixViewMut<T, colmajor> out_l,
		MatrixView<T, colmajor> in_l,
		isize dim,
		T* wp,
		isize j,
		isize offset,
		T p,
		T c,
		T beta) {

	using Info = NativePackInfo<T>;
	constexpr isize N = isize{Info::N};
	constexpr isize N_min = isize{Info::N_min};

	isize const loop_len = dim - (j + 1);
	isize done = loop_len / N * N;
	isize rem = loop_len - done;

	{
		using Pack_ = NativePack<T>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_c = Pack_::broadcast(c);
		Pack_ p_beta = Pack_::broadcast(beta);

		for (isize i = 0; i < done; i += N) {
			detail::rank1_update_inner_loop_packed(
					i, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
		}
	}

#if LDLT_SIMD_HAS_HALF
	if (rem >= (N / 2)) {
		using Pack_ = Pack<T, usize{N / 2}>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_c = Pack_::broadcast(c);
		Pack_ p_beta = Pack_::broadcast(beta);

		detail::rank1_update_inner_loop_packed(
				done, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);

		rem -= N / 2;
		done += N / 2;
	}

#if LDLT_SIMD_HAS_QUARTER

	if (rem >= (N / 4)) {
		using Pack_ = Pack<T, usize{N / 4}>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_c = Pack_::broadcast(c);
		Pack_ p_beta = Pack_::broadcast(beta);

		detail::rank1_update_inner_loop_packed(
				done, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
		rem -= N / 4;
		done += N / 4;
	}

#endif
#endif

	static_assert(N_min == 4 || N_min == 2, ".");

	using Pack_ = Pack<T, 1>;
	Pack_ p_p = {p};
	Pack_ p_c = {c};
	Pack_ p_beta = {beta};

	LDLT_IF_CONSTEXPR(Info::N_min == 4) {
		if (rem >= 2) {
			for (isize i = 0; i < 2; ++i) {
				detail::rank1_update_inner_loop_packed(
						done + i, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
			}
			rem -= 2;
		}
	}
	if (rem == 1) {
		detail::rank1_update_inner_loop_packed(
				done, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
	}
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, false> /*tag*/,
		MatrixViewMut<T, colmajor> out_l,
		MatrixView<T, colmajor> in_l,
		isize dim,
		T* wp,
		isize j,
		isize offset,
		T p,
		T c,
		T beta) {

	using Pack_ = Pack<T, 1>;
	Pack_ p_p = {p};
	Pack_ p_c = {c};
	Pack_ p_beta = {beta};

	isize const loop_len = dim - (j + 1);
	for (isize i = 0; i < loop_len; ++i) {
		detail::rank1_update_inner_loop_packed(
				i, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
	}
}

template <typename T>
LDLT_NO_INLINE void rank1_update(
		LdltViewMut<T> out,
		LdltView<T> in,
		VectorViewMut<T> z,
		isize offset,
		T alpha) {

	isize dim = out.l.rows;
	T* HEDLEY_RESTRICT wp = z.data;

	for (isize j = offset; j < dim; ++j) {
		T p = wp[j - offset];
		T old_dj = in.d(j);
		T new_dj = old_dj + alpha * p * p;
		T gamma = old_dj / new_dj;
		out.d(j) = new_dj;
		T beta = p * alpha / new_dj;
		alpha *= gamma;

		T c = gamma + beta * p;
		out.l(j, j) = T(1);

		detail::rank1_update_inner_loop(
				std::integral_constant<
						bool,
						(std::is_same<T, f32>::value || std::is_same<T, f64>::value)>{},
				out.l,
				in.l,
				dim,
				wp,
				j,
				offset,
				p,
				c,
				beta);
	}
}

template <typename T>
LDLT_NO_INLINE void diagonal_update_multi_pass(
		LdltViewMut<T> out,
		LdltView<T> in,
		VectorView<T> diag_diff,
		isize start_index) {
	isize dim = out.l.rows;
	isize dim_rem = dim - start_index;
	isize idx = start_index;

	bool inplace = out.l.data != in.l.data;

	if (!inplace) {
		if (diag_diff.dim == 0) {
			// copy all of l, d
			out.l.to_eigen() = in.l.to_eigen();
			out.d.to_eigen() = in.d.to_eigen();
		} else {
			// copy left part of l, d
			out.l.block(0, 0, dim, idx).to_eigen() =
					in.l.block(0, 0, dim, idx).to_eigen();
			out.d.segment(0, idx).to_eigen() = in.d.segment(0, idx).to_eigen();
		}
	}
	if (diag_diff.dim == 0) {
		return;
	}

	LdltView<T> current_in = in;

	LDLT_WORKSPACE_MEMORY(ws, dim_rem, T);
	for (isize k = 0; k < diag_diff.dim; ++k) {
		ws[0] = T(1);
		for (isize i = 1; i < dim_rem; ++i) {
			ws[i] = T(0);
		}
		detail::rank1_update( //
				out,
				current_in,
				VectorViewMut<T>{from_ptr_size, ws, dim_rem},
				idx + k,
				diag_diff(k));
		--dim_rem;
		current_in = out.as_const();
	}
}

template <typename T>
struct DiagonalUpdateImpl;

template <>
struct DiagonalUpdateImpl<diagonal_update_strategies::SinglePass> {
	template <typename T>
	LDLT_INLINE static void
	fn(LdltViewMut<T> out,
	   LdltView<T> in,
	   VectorView<T> diag_diff,
	   isize start_index);
};

template <>
struct DiagonalUpdateImpl<diagonal_update_strategies::MultiPass> {
	template <typename T>
	LDLT_INLINE static void
	fn(LdltViewMut<T> out,
	   LdltView<T> in,
	   VectorView<T> diag_diff,
	   isize start_index) {
		detail::diagonal_update_multi_pass(out, in, diag_diff, start_index);
	}
};

template <Layout L>
struct RowAppendImpl;

template <typename T>
LDLT_NO_INLINE void corner_update_impl(
		LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a, T* w) {

	isize dim = in_l.d.dim;

	auto tmp_row = VectorViewMut<T>{from_ptr_size, w, dim}.to_eigen();

	tmp_row = a.segment(0, dim).to_eigen();
	auto l_e = in_l.l.to_eigen();
	l_e.template triangularView<Eigen::UnitLower>().solveInPlace(tmp_row);
	tmp_row.array() /= in_l.d.to_eigen().array();

	{
		auto l = tmp_row.array();
		auto d = in_l.d.to_eigen().array();
		out_l.d(dim) = a(dim) - (l * l * d).sum();
	}

	auto last_row = out_l.l.row(dim);
	if (last_row.data != w) {
		last_row.segment(0, dim).to_eigen() = tmp_row;
		last_row(dim) = T(1);
	}
}

template <>
struct RowAppendImpl<colmajor> {
	template <typename T>
	LDLT_INLINE static void
	corner_update(LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a) {
		isize dim = in_l.d.dim;
		LDLT_WORKSPACE_MEMORY(w, dim, T);
		detail::corner_update_impl(out_l, in_l, a, w);
	}
};

template <typename T>
LDLT_NO_INLINE void
row_append(LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a) {
	isize dim = in_l.d.dim;
	bool inplace = out_l.l.data == in_l.l.data;

	if (!inplace) {
		out_l.l.block(0, 0, dim, dim).to_eigen() = in_l.l.to_eigen();
		out_l.d.segment(0, dim).to_eigen() = in_l.d.to_eigen();
	}
	RowAppendImpl<colmajor>::corner_update(out_l, in_l, a);
	out_l.l(dim, dim) = T(1);
}

template <Layout L>
struct RowDeleteImpl;

template <>
struct RowDeleteImpl<colmajor> {
	template <typename T>
	LDLT_INLINE static void copy_block(
			T* out,
			T const* in,
			isize rows,
			isize cols,
			isize out_outer_stride,
			isize in_outer_stride) {
		for (isize i = 0; i < cols; ++i) {
			T const* in_p = in + (i * in_outer_stride);
			T* out_p = out + (i * out_outer_stride);
			std::copy(in_p, in_p + rows, out_p);
		}
	}

	template <typename T>
	LDLT_NO_INLINE static void handle_bottom_right( //
			LdltViewMut<T> out_bottom_right,
			LdltView<T> in_bottom_right,
			T const* l,
			T d,
			isize rem_dim,
			bool inplace) {

		if (inplace) {
			// const cast is fine here since we can mutate output

			auto l_mut = VectorViewMut<T>{
					from_ptr_size,
					const_cast /* NOLINT */<T*>(l),
					rem_dim,
			};

			// same as in_bottom_right, except mutable
			auto out_from_in = LdltViewMut<T>{
					MatrixViewMut<T, colmajor>{
							from_ptr_rows_cols_stride,
							const_cast /* NOLINT */<T*>(in_bottom_right.l.data),
							in_bottom_right.l.rows,
							in_bottom_right.l.cols,
							in_bottom_right.l.outer_stride,
					},
					VectorViewMut<T>{
							from_ptr_size,
							const_cast /* NOLINT */<T*>(in_bottom_right.d.data),
							in_bottom_right.d.dim,
					},
			};

			detail::rank1_update( //
					out_from_in,
					out_from_in.as_const(),
					l_mut,
					0,
					d);
			// move bottom right corner
			copy_block(
					out_bottom_right.l.data,
					in_bottom_right.l.data,
					rem_dim,
					rem_dim,
					out_bottom_right.l.outer_stride,
					in_bottom_right.l.outer_stride);

			// move bottom part of d
			std::copy(
					in_bottom_right.d.data,
					in_bottom_right.d.data + rem_dim,
					out_bottom_right.d.data);

		} else {
			LDLT_WORKSPACE_MEMORY(w, rem_dim, T);
			std::copy(l, l + rem_dim, w);
			detail::rank1_update( //
					out_bottom_right,
					in_bottom_right,
					VectorViewMut<T>{from_ptr_size, w, rem_dim},
					0,
					d);
		}
	}
};

template <typename T>
LDLT_NO_INLINE void
row_delete_single(LdltViewMut<T> out_l, LdltView<T> in_l, isize i) {
	isize dim = in_l.d.dim;

	bool inplace = out_l.l.data == in_l.l.data;

	// top left
	if (!inplace) {
		out_l.l.block(0, 0, i, i).to_eigen() = in_l.l.block(0, 0, i, i).to_eigen();
		out_l.d.segment(0, i).to_eigen() = in_l.d.segment(0, i).to_eigen();
	}
	if ((i + 1) == dim) {
		return;
	}

	// bottom left
	RowDeleteImpl<colmajor>::copy_block(
			out_l.l.ptr(i, 0),
			in_l.l.ptr(i + 1, 0),
			dim - i,
			i,
			out_l.l.outer_stride,
			in_l.l.outer_stride);

	isize rem_dim = dim - i - 1;
	RowDeleteImpl<colmajor>::handle_bottom_right(
			LdltViewMut<T>{
					out_l.l.block(i, i, rem_dim, rem_dim),
					out_l.d.segment(i, rem_dim),
			},
			LdltView<T>{
					in_l.l.block(i + 1, i + 1, rem_dim, rem_dim),
					in_l.d.segment(i + 1, rem_dim),
			},
			in_l.l.block(i + 1, i, 0, 0).data,
			in_l.d(i),
			rem_dim,
			inplace);
}
extern template void rank1_update(
		LdltViewMut<f32>, LdltView<f32>, VectorViewMut<f32>, isize, f32);
extern template void rank1_update(
		LdltViewMut<f64>, LdltView<f64>, VectorViewMut<f64>, isize, f64);
} // namespace detail

namespace nb {
struct rank1_update {
	template <typename T>
	void operator()(
			LdltViewMut<T> out, LdltView<T> in, VectorView<T> z, T alpha) const {
		isize dim = out.l.rows;
		LDLT_WORKSPACE_MEMORY(wp, out.d.dim, T);
		std::copy(z.data, z.data + dim, wp);
		detail::rank1_update(
				out, in, VectorViewMut<T>{from_ptr_size, wp, dim}, 0, alpha);
	}
};
struct diagonal_update {
	template <typename T, typename S = diagonal_update_strategies::MultiPass>
	LDLT_INLINE void operator()(
			LdltViewMut<T> out,
			LdltView<T> in,
			VectorView<T> diag_diff,
			isize start_index,
			S /*strategy_tag*/ = S{}) const {
		detail::DiagonalUpdateImpl<S>::fn(out, in, diag_diff, start_index);
	}
};

// output.dim == input.dim + 1
struct row_append {
	template <typename T>
	LDLT_INLINE void
	operator()(LdltViewMut<T> out, LdltView<T> in, VectorView<T> new_row) const {
		detail::row_append(out, in, new_row);
	}
};

// output.dim == input.dim - 1
struct row_delete {
	template <typename T>
	LDLT_INLINE void operator()( //
			LdltViewMut<T> out,
			LdltView<T> in,
			isize row_idx) const {
		detail::row_delete_single(out, in, row_idx);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(rank1_update);
LDLT_DEFINE_NIEBLOID(diagonal_update);
LDLT_DEFINE_NIEBLOID(row_append);
LDLT_DEFINE_NIEBLOID(row_delete);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_UPDATE_HPP_OHWFTYRXS */
