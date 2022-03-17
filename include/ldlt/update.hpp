#ifndef INRIA_LDLT_UPDATE_HPP_OHWFTYRXS
#define INRIA_LDLT_UPDATE_HPP_OHWFTYRXS

#include "ldlt/views.hpp"
#include "ldlt/detail/meta.hpp"
#include "veg/memory/dynamic_stack.hpp"

namespace ldlt {

namespace detail {

template <typename T, usize N>
LDLT_INLINE void rank1_update_inner_loop_packed( //
		Pack<T, N> p_p,
		Pack<T, N> p_mu,
		T* inout_l,
		T* wp) {

	Pack<T, N> p_wr = Pack<T, N>::load_unaligned(wp);
	Pack<T, N> p_in_l = Pack<T, N>::load_unaligned(inout_l);

	p_wr = Pack<T, N>::fnmadd(p_p, p_in_l, p_wr);
	p_wr.store_unaligned(wp);
	p_in_l = Pack<T, N>::fmadd(p_mu, p_wr, p_in_l);

	p_in_l.store_unaligned(inout_l);
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::true_type /*tag*/, isize n, T* inout_l, T* wp, T p, T mu) {

	using Info = NativePackInfo<T>;
	constexpr usize N = Info::N;

	isize header_offset =
			detail::bytes_to_next_aligned(inout_l, N * sizeof(T)) / isize{sizeof(T)};

	T* out_l_header_finish = inout_l + min2(header_offset, n);
	T* inout_l_finish = inout_l + n;

	{
		using Pack_ = Pack<T, 1>;
		Pack_ p_p = {p};
		Pack_ p_mu = {mu};

		while (inout_l < out_l_header_finish) {
			detail::rank1_update_inner_loop_packed(p_p, p_mu, inout_l, wp);
			++inout_l;
			++wp;
		}
	}

	isize footer_offset =
			detail::bytes_to_prev_aligned(inout_l_finish, N * sizeof(T)) /
			isize{sizeof(T)};

	T* inout_l_footer_begin = inout_l_finish + footer_offset;
	{
		using Pack_ = NativePack<T>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_mu = Pack_::broadcast(mu);

		while (inout_l < inout_l_footer_begin) {
			detail::rank1_update_inner_loop_packed(p_p, p_mu, inout_l, wp);
			inout_l += N;
			wp += N;
		}
	}
	out_l_header_finish = inout_l_finish;
	{
		using Pack_ = Pack<T, 1>;
		Pack_ p_p = {p};
		Pack_ p_mu = {mu};

		while (inout_l < out_l_header_finish) {
			detail::rank1_update_inner_loop_packed(p_p, p_mu, inout_l, wp);
			++inout_l;
			++wp;
		}
	}
}

template <typename T>
LDLT_INLINE void rank1_update_inner_loop(
		std::false_type /*tag*/, isize dim, T* inout_l, T* wp, T p, T mu) {

	using Pack_ = Pack<T, 1>;
	Pack_ p_p = {p};
	Pack_ p_mu = {mu};

	for (isize i = 0; i < dim; ++i) {
		detail::rank1_update_inner_loop_packed(p_p, p_mu, inout_l, wp);
	}
}

template <typename T>
LDLT_NO_INLINE void
rank1_update_clobber_z(LdltViewMut<T> inout, VectorViewMut<T> z, T alpha) {

	isize dim = inout.d().dim;
	T* wp = z.data;

	for (isize j = 0; j < dim; ++j) {
		T p = *wp;
		++wp;
		T old_dj = inout.d()(j);
		T new_dj = old_dj + alpha * p * p;
		T mu = alpha * p / new_dj;
		alpha -= new_dj * mu * mu;

		isize tail_dim = dim - j - 1;
		detail::rank1_update_inner_loop(
				should_vectorize<T>{},
				tail_dim,
				inout.l_mut().col(j).segment(j + 1, tail_dim).data,
				wp,
				p,
				mu);

		inout.d_mut()(j) = new_dj;
	}
}

template <Layout L>
struct RowAppendImpl;

template <typename T>
LDLT_NO_INLINE void corner_update_impl(
		LdltViewMut<T> out_l,
		LdltView<T> in_l,
		VectorView<T> a,
		VectorViewMut<T> w) {

	isize dim = in_l.d().dim;

	auto new_row = w.to_eigen();
	new_row = a.segment(0, dim).to_eigen();
	auto l_e = in_l.l().to_eigen();
	l_e.template triangularView<Eigen::UnitLower>().solveInPlace(new_row);
	new_row = new_row.cwiseQuotient(in_l.d().to_eigen());

	{
		auto l = new_row;
		auto d = in_l.d().to_eigen();
		out_l.d_mut()(dim) = a(dim) - (l.cwiseProduct(l).cwiseProduct(d)).sum();
	}

	auto last_row = out_l.l_mut().row(dim);
	if (last_row.data != w.data) {
		last_row.segment(0, dim).to_eigen() = new_row;
	}
}

template <>
struct RowAppendImpl<colmajor> {
	template <typename T>
	LDLT_INLINE static void
	corner_update(LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a) {
		isize dim = in_l.d().dim;
		auto ws = out_l.l_mut().col(dim).segment(0, dim);
		detail::corner_update_impl(out_l, in_l, a, ws);
	}
};

template <typename T>
LDLT_NO_INLINE void
row_append(LdltViewMut<T> out_l, LdltView<T> in_l, VectorView<T> a) {
	isize n = in_l.d().dim;
	VEG_ASSERT(out_l.d().dim == (n + 1));
	bool inplace = out_l.l().data == in_l.l().data;

	if (!inplace) {
		out_l.l_mut().block(0, 0, n, n).to_eigen() = in_l.l().to_eigen();
	}
	RowAppendImpl<colmajor>::corner_update(out_l, in_l, a);
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
			isize rem_dim) {

		// const cast is fine here since we can mutate output

		auto l_mut = VectorViewMut<T>{
				from_ptr_size,
				const_cast /* NOLINT */<T*>(l),
				rem_dim,
		};

		// same as in_bottom_right, except mutable
		auto out_from_in = LdltViewMut<T>{{
				from_ptr_rows_cols_stride,
				const_cast /* NOLINT */<T*>(in_bottom_right.l().data),
				in_bottom_right.l().rows,
				in_bottom_right.l().cols,
				in_bottom_right.l().outer_stride,
		}};

		detail::rank1_update_clobber_z( //
				out_from_in,
				l_mut,
				d);
		// move bottom right corner
		copy_block(
				out_bottom_right.l_mut().data,
				in_bottom_right.l().data,
				rem_dim,
				rem_dim,
				out_bottom_right.l().outer_stride,
				in_bottom_right.l().outer_stride);
	}
};

template <typename T>
LDLT_NO_INLINE void row_delete_single(LdltViewMut<T> inout, isize i) {
	isize n = inout.d().dim;
	VEG_ASSERT(n >= 1);
	auto out = inout.head(n - 1);
	auto in = inout.as_const();

	bool inplace = out.l().data == in.l().data;
	VEG_ASSERT(inplace);

	// top left
	if ((i + 1) == n) {
		return;
	}

	// bottom left
	RowDeleteImpl<colmajor>::copy_block(
			out.l_mut().ptr(i, 0),
			in.l().ptr(i + 1, 0),
			n - i,
			i,
			out.l().outer_stride,
			in.l().outer_stride);

	isize rem_dim = n - i - 1;
	RowDeleteImpl<colmajor>::handle_bottom_right(
			out.tail(rem_dim),
			in.tail(rem_dim),
			in.l().ptr(i + 1, i),
			in.d()(i),
			rem_dim);
}

} // namespace detail

namespace nb {
struct rank1_update_req {
	template <typename T>
	LDLT_INLINE auto operator()(veg::Tag<T> /*tag*/, isize n) const
			-> veg::dynstack::StackReq {
		return {isize(sizeof(T)) * n, detail::_align<T>()};
	}
};

struct rank1_update {
	template <typename T>
	LDLT_INLINE void operator()(
			LdltViewMut<T> inout,
			VectorView<T> z,
			T alpha,
			veg::dynstack::DynStackMut stack) const {
		isize n = inout.d().dim;
		auto _ = stack.make_new_for_overwrite(veg::Tag<T>{}, n, detail::_align<T>())
		             .unwrap();
		std::copy(z.data, z.data + n, _.ptr_mut());
		detail::rank1_update_clobber_z(
				inout, {from_ptr_size, _.ptr_mut(), n}, alpha);
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
			LdltViewMut<T> inout,
			isize row_idx) const {
		detail::row_delete_single(inout, row_idx);
	}
};
} // namespace nb
LDLT_DEFINE_NIEBLOID(rank1_update_req);
LDLT_DEFINE_NIEBLOID(rank1_update);
LDLT_DEFINE_NIEBLOID(row_append);
LDLT_DEFINE_NIEBLOID(row_delete);
} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_UPDATE_HPP_OHWFTYRXS */
