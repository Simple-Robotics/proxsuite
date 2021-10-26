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
		Pack<T, N> p_p,
		Pack<T, N> p_c,
		Pack<T, N> p_beta,

		T* out_l,
		T const* in_l,
		T* wp) {

	Pack<T, N> p_wr = Pack<T, N>::load_unaligned(wp);
	Pack<T, N> p_in_l = Pack<T, N>::load_unaligned(in_l);

	p_wr = Pack<T, N>::fnmadd(p_p, p_in_l, p_wr);
	p_wr.store_unaligned(wp);
	p_in_l = p_in_l.mul(p_c);
	p_in_l = Pack<T, N>::fmadd(p_beta, p_wr, p_in_l);

	p_in_l.store_unaligned(out_l);
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, true> /*tag*/,
		isize dim,
		T* out_l,
		T const* in_l,
		T* wp,
		T p,
		T c,
		T beta) {

	using Info = NativePackInfo<T>;
	constexpr usize N = Info::N;

	isize header_offset =
			detail::bytes_to_next_aligned(out_l, N * sizeof(T)) / isize{sizeof(T)};

	T* out_l_header_finish = out_l + min2(header_offset, dim);
	T* out_l_finish = out_l + dim;

	bool early_exit = out_l_header_finish == out_l_finish;

	{
	scalar_loop:
		using Pack_ = Pack<T, 1>;
		Pack_ p_p = {p};
		Pack_ p_c = {c};
		Pack_ p_beta = {beta};

		while (out_l < out_l_header_finish) {
			detail::rank1_update_inner_loop_packed(p_p, p_c, p_beta, out_l, in_l, wp);
			++out_l;
			++in_l;
			++wp;
		}
	}
	if (early_exit) {
		return;
	}

	isize footer_offset =
			detail::bytes_to_prev_aligned(out_l_finish, N * sizeof(T)) /
			isize{sizeof(T)};

	T* out_l_footer_begin = out_l_finish + footer_offset;
	{
		using Pack_ = NativePack<T>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_c = Pack_::broadcast(c);
		Pack_ p_beta = Pack_::broadcast(beta);

		while (out_l < out_l_footer_begin) {
			detail::rank1_update_inner_loop_packed(p_p, p_c, p_beta, out_l, in_l, wp);
			out_l += N;
			in_l += N;
			wp += N;
		}
	}
	out_l_header_finish = out_l_finish;
	early_exit = true;
	goto scalar_loop; // NOLINT(hicpp-avoid-goto, cppcoreguidelines-avoid-goto)
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, false> /*tag*/,
		isize dim,
		T* out_l,
		T const* in_l,
		T* wp,
		T p,
		T c,
		T beta) {

	using Pack_ = Pack<T, 1>;
	Pack_ p_p = {p};
	Pack_ p_c = {c};
	Pack_ p_beta = {beta};

	for (isize i = 0; i < dim; ++i) {
		detail::rank1_update_inner_loop_packed(p_p, p_c, p_beta, out_l, in_l, wp);
	}
}

template <typename T>
LDLT_NO_INLINE void
rank1_update(LdltViewMut<T> out, LdltView<T> in, VectorViewMut<T> z, T alpha) {

	isize dim = out.l.rows;
	T* wp = z.data;

	for (isize j = 0; j < dim; ++j) {
		T p = *wp;
		++wp;
		T old_dj = in.d(j);
		T new_dj = old_dj + alpha * p * p;
		T gamma = old_dj / new_dj;
		out.d(j) = new_dj;
		T beta = p * alpha / new_dj;
		alpha *= gamma;

		T c = gamma + beta * p;
		out.l(j, j) = T(1);

		isize tail_dim = dim - j - 1;
		detail::rank1_update_inner_loop(
				std::integral_constant<
						bool,
						(std::is_same<T, f32>::value || std::is_same<T, f64>::value)>{},
				tail_dim,
				out.l.col(j).segment(j + 1, tail_dim).data,
				in.l.col(j).segment(j + 1, tail_dim).data,
				wp,
				p,
				c,
				beta);
	}
}

template <usize N, usize Unroll = 1, typename T>
LDLT_INLINE void rank_r_update_inner_loop_packed(
		isize r,
		T const* HEDLEY_RESTRICT ptr_p,
		T const* HEDLEY_RESTRICT ptr_c,
		T const* HEDLEY_RESTRICT ptr_b,
		T* HEDLEY_RESTRICT out_l,
		T const* HEDLEY_RESTRICT in_l,
		T* HEDLEY_RESTRICT wp,
		isize wp_stride) {

	Pack<T, N> p_in_l[Unroll];

	for (usize u = 0; u < Unroll; ++u) {
		p_in_l[u] = Pack<T, N>::load_unaligned(in_l + N * u);
	}

	T* wp_finish = wp + r * wp_stride;
	while (wp < wp_finish) {
		// registers used:
		// - with fma: UNROLL + 3 + 1
		// - w/o  fma: UNROLL + 3 + 1 + 1
		//
		// K + 2×UNROLL
		// UNROLL = NREG - K
		//
		// w/o  fma   : 11
		// with fma   : 12
		// with avx512: 28
		Pack<T, N> p_p = Pack<T, N>::load_unaligned(ptr_p);
		Pack<T, N> p_c = Pack<T, N>::load_unaligned(ptr_c);
		Pack<T, N> p_b = Pack<T, N>::load_unaligned(ptr_b);

		for (usize u = 0; u < Unroll; ++u) {
			Pack<T, N> p_wr = Pack<T, N>::load_unaligned(wp + N * u);
			p_wr = Pack<T, N>::fnmadd(p_p, p_in_l[u], p_wr);
			p_wr.store_unaligned(wp + N * u);

			p_in_l[u] = p_in_l[u].mul(p_c);
			p_in_l[u] = Pack<T, N>::fmadd(p_b, p_wr, p_in_l[u]);
		}

		wp += wp_stride;
		ptr_p += N;
		ptr_c += N;
		ptr_b += N;
	}

	for (usize u = 0; u < Unroll; ++u) {
		p_in_l[u].store_unaligned(out_l + N * u);
	}
}

template <typename T>
LDLT_INLINE void rank_r_update_inner_loop( //
		std::integral_constant<bool, true> /*tag*/,
		isize dim,
		isize r,

		T* out_l,
		T const* in_l,
		T* wp,
		isize wp_stride,

		T const* scalar_p,
		T const* scalar_c,
		T const* scalar_b,

		T const* vector_p,
		T const* vector_c,
		T const* vector_b) {

	(void)vector_p, (void)vector_c, (void)vector_b;

	using Info = NativePackInfo<T>;
	constexpr usize N = Info::N;
	isize header_offset =
			detail::bytes_to_next_aligned(out_l, N * sizeof(T)) / isize{sizeof(T)};
	T* out_l_header_finish = out_l + min2(header_offset, dim);
	T* out_l_finish = out_l + dim;
	bool early_exit = out_l_header_finish == out_l_finish;

	{
	scalar_loop:
		while (out_l < out_l_header_finish) {
			detail::rank_r_update_inner_loop_packed<1>( //
					r,
					scalar_p,
					scalar_c,
					scalar_b,
					out_l,
					in_l,
					wp,
					wp_stride);
			++out_l;
			++in_l;
			++wp;
		}
	}
	if (early_exit) {
		return;
	}
	isize footer_offset =
			detail::bytes_to_prev_aligned(out_l_finish, N * sizeof(T)) /
			isize{sizeof(T)};
	T* out_l_footer_begin = out_l_finish + footer_offset;

	{
		constexpr usize Unroll = 10;

		isize n_unroll =
				(out_l_footer_begin - out_l) / isize(N * Unroll) * isize(N * Unroll);
		T* out_l_unroll_finish = out_l + n_unroll;
		while (out_l < out_l_unroll_finish) {
			detail::rank_r_update_inner_loop_packed<N, Unroll>( //
					r,
					vector_p,
					vector_c,
					vector_b,
					out_l,
					in_l,
					wp,
					wp_stride);

			out_l += N * Unroll;
			in_l += N * Unroll;
			wp += N * Unroll;
		}
	}
	{
		constexpr usize Unroll = 4;

		isize n_unroll =
				(out_l_footer_begin - out_l) / isize(N * Unroll) * isize(N * Unroll);
		T* out_l_unroll_finish = out_l + n_unroll;
		while (out_l < out_l_unroll_finish) {
			detail::rank_r_update_inner_loop_packed<N, Unroll>( //
					r,
					vector_p,
					vector_c,
					vector_b,
					out_l,
					in_l,
					wp,
					wp_stride);

			out_l += N * Unroll;
			in_l += N * Unroll;
			wp += N * Unroll;
		}
	}
	{
		constexpr usize Unroll = 2;

		isize n_unroll =
				(out_l_footer_begin - out_l) / isize(N * Unroll) * isize(N * Unroll);
		T* out_l_unroll_finish = out_l + n_unroll;
		while (out_l < out_l_unroll_finish) {
			detail::rank_r_update_inner_loop_packed<N, Unroll>( //
					r,
					vector_p,
					vector_c,
					vector_b,
					out_l,
					in_l,
					wp,
					wp_stride);

			out_l += N * Unroll;
			in_l += N * Unroll;
			wp += N * Unroll;
		}
	}

	{
		while (out_l < out_l_footer_begin) {
			detail::rank_r_update_inner_loop_packed<N>( //
					r,
					vector_p,
					vector_c,
					vector_b,
					out_l,
					in_l,
					wp,
					wp_stride);

			out_l += N;
			in_l += N;
			wp += N;
		}
	}
	out_l_header_finish = out_l_finish;
	early_exit = true;
	goto scalar_loop; // NOLINT(hicpp-avoid-goto, cppcoreguidelines-avoid-goto)
}

template <typename T>
LDLT_INLINE void rank_r_update_inner_loop( //
		std::integral_constant<bool, false> /*tag*/,
		isize dim,
		isize r,

		T* out_l,
		T const* in_l,
		T* wp,
		isize wp_stride,

		T const* scalar_p,
		T const* scalar_c,
		T const* scalar_b,

		T const* vector_p,
		T const* vector_c,
		T const* vector_b) {
	(void)vector_p, (void)vector_c, (void)vector_b;

	for (isize i = 0; i < dim; ++i) {
		detail::rank_r_update_inner_loop_packed<1>( //
				r,
				scalar_p,
				scalar_c,
				scalar_b,
				out_l + i,
				in_l + i,
				wp + i,
				wp_stride);
	}
}

template <typename T>
LDLT_NO_INLINE void rank_r_update( //
		LdltViewMut<T> out,
		LdltView<T> in,
		MatrixViewMut<T, colmajor> z,
		VectorViewMut<T> alpha) {

	using Info = NativePackInfo<T>;
	constexpr usize N = Info::N;

	isize n = out.l.rows;
	isize r = z.cols;
	T* wp = z.data;
	isize wp_stride = z.outer_stride;

	LDLT_MULTI_WORKSPACE_MEMORY(
			(ps, Uninit, Vec(r), alignof(T), T), //
			(cs, Uninit, Vec(r), alignof(T), T),
			(bs, Uninit, Vec(r), alignof(T), T),

			(packed_ps, Uninit, Mat(N, r), N * alignof(T), T), //
			(packed_cs, Uninit, Mat(N, r), N * alignof(T), T),
			(packed_bs, Uninit, Mat(N, r), N * alignof(T), T));

	isize j = 0;
	while (true) {
		T old_dj = in.d(j);
		for (isize i = 0; i < r; ++i) {
			T p = wp[i * wp_stride];

			T new_dj = old_dj + alpha(i) * p * p;

			T gamma = old_dj / new_dj;

			T b = p * alpha(i) / new_dj;
			T c = gamma + b * p;

			old_dj = new_dj;

			alpha(i) *= gamma;
			out.l(j, j) = T(1);

			ps(i) = p;
			bs(i) = b;
			cs(i) = c;
			Pack<T, N>::broadcast(p).store_unaligned(packed_ps.col(i).data);
			Pack<T, N>::broadcast(b).store_unaligned(packed_bs.col(i).data);
			Pack<T, N>::broadcast(c).store_unaligned(packed_cs.col(i).data);
		}
		out.d(j) = old_dj;

		isize tail_dim = n - j - 1;
		if (tail_dim == 0) {
			break;
		}
		++wp;
		detail::rank_r_update_inner_loop(
				std::integral_constant<
						bool,
						(std::is_same<T, f32>::value || std::is_same<T, f64>::value)>{},
				tail_dim,
				r,

				out.l.col(j).segment(j + 1, tail_dim).data,
				in.l.col(j).segment(j + 1, tail_dim).data,
				wp,
				wp_stride,

				ps.data,
				cs.data,
				bs.data,

				packed_ps.data,
				packed_cs.data,
				packed_bs.data);
		++j;
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

	// set bottom right element to zero for use in workspace
	out.l(dim - 1, dim - 1) = 0;
	for (isize k = 0; k < diag_diff.dim; ++k) {
		auto ws = out.l.col(dim - 1).segment(0, dim_rem);
		// ws is already zeroed
		auto tail_size = dim - (idx + k);

		ws(0) = T(1);
		detail::rank1_update( //
				out.tail(tail_size),
				current_in.tail(tail_size),
				ws,
				diag_diff(k));
		--dim_rem;

		detail::set_zero(ws.data, usize(ws.dim));
		current_in = out.as_const();
	}
	// restore bottom right element
	out.l(dim - 1, dim - 1) = 1;
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
		LdltViewMut<T> out_l,
		LdltView<T> in_l,
		VectorView<T> a,
		VectorViewMut<T> w) {

	isize dim = in_l.d.dim;

	auto new_row = w.to_eigen();
	new_row = a.segment(0, dim).to_eigen();
	auto l_e = in_l.l.to_eigen();
	l_e.template triangularView<Eigen::UnitLower>().solveInPlace(new_row);
	new_row = new_row.cwiseQuotient(in_l.d.to_eigen());

	{
		auto l = new_row;
		auto d = in_l.d.to_eigen();
		out_l.d(dim) = a(dim) - (l.cwiseProduct(l).cwiseProduct(d)).sum();
	}

	auto last_row = out_l.l.row(dim);
	if (last_row.data != w.data) {
		last_row.segment(0, dim).to_eigen() = new_row;
		last_row(dim) = T(1);
	}
}

template <>
struct RowAppendImpl<colmajor> {
	template <typename T>
	LDLT_INLINE static void
	corner_update(LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a) {
		isize dim = in_l.d.dim;
		auto ws = out_l.l.col(dim).segment(0, dim);
		detail::corner_update_impl(out_l, in_l, a, ws);
		detail::set_zero(ws.data, usize(dim));
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
			auto w = out_bottom_right.l.col(rem_dim - 1);
			std::copy(l, l + rem_dim, w.data);
			detail::rank1_update( //
					out_bottom_right,
					in_bottom_right,
					w,
					d);
			detail::set_zero(w.data, usize(w.dim));
			w(rem_dim - 1) = 1;
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
LDLT_EXPLICIT_TPL_DECL(4, rank1_update<f32>);
LDLT_EXPLICIT_TPL_DECL(4, rank1_update<f64>);
} // namespace detail

namespace nb {
struct rank1_update {
	template <typename T>
	LDLT_INLINE void operator()(
			LdltViewMut<T> out, LdltView<T> in, VectorView<T> z, T alpha) const {
		isize dim = out.l.rows;

		auto wp = out.l.col(dim - 1);
		std::copy(z.data, z.data + dim, wp.data);
		detail::rank1_update(out, in, wp, alpha);
		detail::set_zero(wp.data, usize(dim - 1));
		wp(dim - 1) = 1;
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
