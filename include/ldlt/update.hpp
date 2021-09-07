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

template <typename Scalar, Layout L, usize N>
LDLT_INLINE void rank1_update_inner_loop_packed(
		i32 i,
		Pack<Scalar, N> p_p,
		Pack<Scalar, N> p_c,
		Pack<Scalar, N> p_beta,

		MatrixViewMut<Scalar, L> out_l,
		MatrixView<Scalar, L> in_l,
		Scalar* wp,
		i32 j,
		i32 offset) {
	i32 r = i + j + 1;

	// TODO[PERF]: check asm, clang does weird stuff with address computations in
	// tight loop

	auto in_l_ptr = ElementAccess<L>::offset(in_l.data, r, j, in_l.outer_stride);
	auto out_l_ptr =
			ElementAccess<L>::offset(out_l.data, r, j, out_l.outer_stride);

	Pack<Scalar, N> p_wr = Pack<Scalar, N>::load_unaligned(wp + (r - offset));
	Pack<Scalar, N> p_in_l =
			ElementAccess<L>::template load_col_pack<N>(in_l_ptr, in_l.outer_stride);

	p_wr = Pack<Scalar, N>::fnmadd(p_p, p_in_l, p_wr);
	p_wr.store_unaligned(wp + (r - offset));
	p_in_l = p_in_l.mul(p_c);
	p_in_l = Pack<Scalar, N>::fmadd(p_beta, p_wr, p_in_l);

	ElementAccess<L>::store_col_pack(out_l_ptr, p_in_l, out_l.outer_stride);
}

template <typename Scalar, Layout L>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, true> /*tag*/,
		MatrixViewMut<Scalar, L> out_l,
		MatrixView<Scalar, L> in_l,
		i32 dim,
		Scalar* wp,
		i32 j,
		i32 offset,
		Scalar p,
		Scalar c,
		Scalar beta) {

	using Info = NativePackInfo<Scalar>;
	constexpr i32 N = i32{Info::N};
	constexpr i32 N_min = i32{Info::N_min};

	i32 const loop_len = dim - (j + 1);
	i32 done = loop_len / N * N;
	i32 rem = loop_len - done;

	{
		using Pack_ = NativePack<Scalar>;
		Pack_ p_p = Pack_::broadcast(p);
		Pack_ p_c = Pack_::broadcast(c);
		Pack_ p_beta = Pack_::broadcast(beta);

		for (i32 i = 0; i < done; i += N) {
			detail::rank1_update_inner_loop_packed(
					i, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
		}
	}

#if LDLT_SIMD_HAS_HALF
	if (rem >= (N / 2)) {
		using Pack_ = Pack<Scalar, usize{N / 2}>;
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
		using Pack_ = Pack<Scalar, usize{N / 4}>;
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

	using Pack_ = Pack<Scalar, 1>;
	Pack_ p_p = {p};
	Pack_ p_c = {c};
	Pack_ p_beta = {beta};

	LDLT_IF_CONSTEXPR(Info::N_min == 4) {
		if (rem >= 2) {
			for (i32 i = 0; i < 2; ++i) {
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

template <typename Scalar, Layout L>
LDLT_INLINE void rank1_update_inner_loop(
		std::integral_constant<bool, false> /*tag*/,
		MatrixViewMut<Scalar, L> out_l,
		MatrixView<Scalar, L> in_l,
		i32 dim,
		Scalar* wp,
		i32 j,
		i32 offset,
		Scalar p,
		Scalar c,
		Scalar beta) {

	using Pack_ = Pack<Scalar, 1>;
	Pack_ p_p = {p};
	Pack_ p_c = {c};
	Pack_ p_beta = {beta};

	i32 const loop_len = dim - (j + 1);
	for (i32 i = 0; i < loop_len; ++i) {
		detail::rank1_update_inner_loop_packed(
				i, p_p, p_c, p_beta, out_l, in_l, wp, j, offset);
	}
}

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
		out.l(j, j) = Scalar(1);

		detail::rank1_update_inner_loop(
				std::integral_constant<
						bool,
						(std::is_same<Scalar, f32>::value ||
		         std::is_same<Scalar, f64>::value)>{},
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

struct Dyn {
	i32 inner;
	LDLT_INLINE constexpr auto value() const noexcept -> i32 { return inner; }
	LDLT_INLINE constexpr auto incr() const noexcept -> Dyn {
		return {inner + 1};
	}
	LDLT_INLINE constexpr auto decr() const noexcept -> Dyn {
		return {inner - 1};
	}

	template <typename Fn>
	LDLT_INLINE void loop(Fn fn) const {
		for (i32 i = 0; i < inner; ++i) {
			fn(Dyn{i});
		}
	}
};

struct Empty {};

template <i32 N>
struct Fix {
	LDLT_INLINE constexpr auto value() const noexcept -> i32 { return N; }
	LDLT_INLINE constexpr auto incr() const noexcept -> Fix<N + 1> { return {}; }
	LDLT_INLINE constexpr auto decr() const noexcept -> Fix<N - 1> { return {}; }

	template <typename Fn, i32... Is>
	LDLT_INLINE void
	loop_impl(Fn fn, integer_sequence<i32, Is...> /*tag*/) const {
		using Arr = Empty[];
		(void)Arr{Empty{}, (void(fn(Fix<Is>{})), Empty{})...};
	}

	template <typename Fn>
	LDLT_INLINE void loop(Fn fn) const {
		loop_impl(LDLT_FWD(fn), make_integer_sequence<i32, N>{});
	}

	LDLT_INLINE operator Dyn /* NOLINT */() const noexcept { return {N}; }
};

template <typename Scalar>
struct LoopData {
	Scalar* ws;
	Scalar* ps;
	Scalar* betas;
	Scalar* cs;
	i32 start_index;
	i32 dim_rest;
};

template <typename Scalar>
struct InnerLoop {
	Scalar& lr;
	i32& w_offset;
	LoopData<Scalar> data;
	i32 r;

	LDLT_INLINE void operator()(Dyn k) const {
		LDLT_FP_PRAGMA
		auto& wr = data.ws[w_offset + (r - data.start_index)];
		wr = wr - data.ps[k.value()] * lr;
		lr = data.cs[k.value()] * lr + data.betas[k.value()] * wr;

		w_offset += data.dim_rest;
	}
};

template <typename Scalar, Layout L>
struct OuterLoop {
	MatrixViewMut<Scalar, L> out_l;
	MatrixView<Scalar, L> in_l;
	LoopData<Scalar> data;
	i32 j;

	LDLT_INLINE void operator()(Fix<0> /*n_diag*/) const {}

	template <typename RMinusJ>
	LDLT_INLINE void operator()(RMinusJ r_minus_j) const {
		if (r_minus_j.value() > 0) {
			i32 r = r_minus_j.value() + j;
			Scalar lr = in_l(r, j);
			i32 w_offset = 0;
			r_minus_j.loop(InnerLoop<Scalar>{lr, w_offset, data, r});
			out_l(r, j) = lr;
		}
	}
};

template <typename Scalar, Layout L, typename NDiag>
LDLT_NO_INLINE void diagonal_update_single_pass(
		LdltViewMut<Scalar, L> out,
		LdltView<Scalar, L> in,
		VectorView<Scalar> diag_diff,
		i32 start_index,
		NDiag n_diag) {
	// FIXME: buggy for ndiag > 1

	i32 dim = out.l.rows;
	i32 n_diag_terms = n_diag.value();
	if (n_diag_terms == 0) {
		return;
	}
	i32 dim_rest = dim - start_index;

	LDLT_MULTI_WORKSPACE_MEMORY(
			((ws, n_diag_terms * dim_rest),
	     (alphas, n_diag_terms),
	     (ps, n_diag_terms),
	     (betas, n_diag_terms),
	     (cs, n_diag_terms)),
			Scalar);

	for (i32 k = 0; k < n_diag_terms * dim_rest; ++k) {
		ws[k] = Scalar(0);
	}
	for (i32 k = 0; k < n_diag_terms; ++k) {
		ws[k * dim_rest + k] = Scalar(1);
		alphas[k] = diag_diff(k);
	}

	for (i32 j = start_index; j < dim; ++j) {

		{
			Scalar dj = in.d(j);
			for (i32 k = 0, w_offset = 0; //
			     k < n_diag_terms;
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

		if (j + n_diag_terms <= dim) {
			n_diag.loop(OuterLoop<Scalar, L>{
					out.l,
					in.l,
					{ws, ps, betas, cs, start_index, dim_rest},
					j,
			});
		} else {
			Dyn{dim - j}.loop(OuterLoop<Scalar, L>{
					out.l,
					in.l,
					{ws, ps, betas, cs, start_index, dim_rest},
					j,
			});
		}

		// TODO: vectorize
		for (i32 r = j + n_diag_terms; r < dim; ++r) {
			i32 w_offset = 0;
			Scalar lr = in.l(r, j);
			n_diag.loop(InnerLoop<Scalar>{
					lr,
					w_offset,
					{ws, ps, betas, cs, start_index, dim_rest},
					r,
			});
			out.l(r, j) = lr;
		}
	}
}

template <typename Scalar, Layout L>
LDLT_INLINE void diagonal_update_multi_pass(
		LdltViewMut<Scalar, L> out,
		LdltView<Scalar, L> in,
		VectorView<Scalar> diag_diff,
		i32 start_index) {
	i32 dim = out.l.rows;
	i32 dim_rem = dim - start_index;
	LDLT_WORKSPACE_MEMORY(ws, dim_rem, Scalar);
	for (i32 k = 0; k < diag_diff.dim; ++k) {
		ws[0] = Scalar(1);
		for (i32 i = 1; i < dim_rem; ++i) {
			ws[i] = Scalar(0);
		}
		detail::rank1_update( //
				out,
				in,
				VectorViewMut<Scalar>{ws, dim_rem},
				start_index + k,
				diag_diff(k));
		--dim_rem;
	}
}

template <typename T>
struct DiagonalUpdateImpl;

template <>
struct DiagonalUpdateImpl<diagonal_update_strategies::SinglePass> {
	template <typename Scalar, Layout L>
	LDLT_INLINE static void
	fn(LdltViewMut<Scalar, L> out,
	   LdltView<Scalar, L> in,
	   VectorView<Scalar> diag_diff,
	   i32 start_index) {
		detail::diagonal_update_single_pass(
				out, in, diag_diff, start_index, Dyn{diag_diff.dim});
	}
};

template <>
struct DiagonalUpdateImpl<diagonal_update_strategies::MultiPass> {
	template <typename Scalar, Layout L>
	LDLT_INLINE static void
	fn(LdltViewMut<Scalar, L> out,
	   LdltView<Scalar, L> in,
	   VectorView<Scalar> diag_diff,
	   i32 start_index) {
		detail::diagonal_update_multi_pass(out, in, diag_diff, start_index);
	}
};

template <Layout L>
struct RowAppendImpl;

template <typename T, Layout L>
LDLT_NO_INLINE void corner_update_impl(
		LdltViewMut<T, L> out_l, LdltView<T, L> in_l, VectorView<T> a, T* w) {

	i32 dim = in_l.d.dim;

	auto tmp_row = to_eigen_vector_mut(VectorViewMut<T>{w, dim});

	tmp_row = to_eigen_vector(a.segment(0, dim));
	auto l_e = to_eigen_matrix(in_l.l);
	l_e.template triangularView<Eigen::UnitLower>().solveInPlace(tmp_row);
	tmp_row.array() /= to_eigen_vector(in_l.d).array();

	{
		auto l = tmp_row.array();
		auto d = to_eigen_vector(in_l.d).array();
		out_l.d(dim) = a(dim) - (l * l * d).sum();
	}

	auto last_row = out_l.l.row(dim);
	if (last_row.data != w) {
		to_eigen_vector_mut(out_l.l.row(dim)) = tmp_row;
	}
}

template <>
struct RowAppendImpl<rowmajor> {
	template <typename T>
	LDLT_INLINE static void corner_update(
			LdltViewMut<T, rowmajor> out_l,
			LdltView<T, rowmajor> in_l,
			VectorView<T> a) {
		i32 dim = in_l.d.dim;
		detail::corner_update_impl(out_l, in_l, a, out_l.l.row(dim));
	}
};
template <>
struct RowAppendImpl<colmajor> {
	template <typename T>
	LDLT_NO_INLINE static void corner_update(
			LdltViewMut<T, colmajor> out_l,
			LdltView<T, colmajor> in_l,
			VectorView<T> a) {
		i32 dim = in_l.d.dim;
		LDLT_WORKSPACE_MEMORY(w, dim, T);
		detail::corner_update_impl(out_l, in_l, a, w);
	}
};

template <typename T, Layout L>
LDLT_NO_INLINE void
row_append(LdltViewMut<T, L> out_l, LdltView<T, L> in_l, VectorView<T> a) {
	i32 dim = in_l.d.dim;
	bool inplace = out_l.l.data == in_l.l.data;

	if (!inplace) {
		to_eigen_matrix_mut(out_l.l.block(0, 0, dim, dim)) =
				to_eigen_matrix(in_l.l);
	}
	RowAppendImpl<L>::corner_update(out_l, in_l, a);
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
			i32 rows,
			i32 cols,
			i32 out_outer_stride,
			i32 in_outer_stride) {
		for (i32 i = 0; i < cols; ++i) {
			T const* in_p = in + (i * in_outer_stride);
			T* out_p = out + (i * out_outer_stride);
			std::copy(in_p, in_p + rows, out_p);
		}
	}

	template <typename T>
	LDLT_NO_INLINE static void handle_bottom_right( //
			LdltViewMut<T, colmajor> out_bottom_right,
			LdltView<T, colmajor> in_bottom_right,
			T const* l,
			T d,
			i32 rem_dim,
			bool inplace) {

		if (inplace) {
			// const cast is fine here since we can mutate output

			auto l_mut = VectorViewMut<T>{
					const_cast /* NOLINT */<T*>(l),
					rem_dim,
			};

			// same as in_bottom_right, except mutable
			auto out_from_in = LdltViewMut<T, colmajor>{
					MatrixViewMut<T, colmajor>{
							const_cast /* NOLINT */<T*>(in_bottom_right.l.data),
							in_bottom_right.l.rows,
							in_bottom_right.l.cols,
							in_bottom_right.l.outer_stride,
					},
					VectorViewMut<T>{
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
					VectorViewMut<T>{w, rem_dim},
					0,
					d);
		}
	}
};

template <>
struct RowDeleteImpl<rowmajor> {
	template <typename T>
	LDLT_INLINE static void copy_block(
			T* out,
			T const* in,
			i32 rows,
			i32 cols,
			i32 out_outer_stride,
			i32 in_outer_stride) {
		for (i32 i = 0; i < rows; ++i) {
			T const* in_p = in + (i * in_outer_stride);
			T* out_p = out + (i * out_outer_stride);
			std::copy(in_p, in_p + cols, out_p);
		}
	}

	template <typename T>
	LDLT_NO_INLINE static void handle_bottom_right( //
			LdltViewMut<T, rowmajor> out_bottom_right,
			LdltView<T, rowmajor> in_bottom_right,
			T const* l,
			T d,
			i32 rem_dim,
			bool /*inplace*/) {

		LDLT_WORKSPACE_MEMORY(w, rem_dim, T);
		for (i32 i = 0; i < rem_dim; ++i) {
			w[i] = l[i * in_bottom_right.l.outer_stride];
		}
		detail::rank1_update( //
				out_bottom_right,
				in_bottom_right,
				VectorViewMut<T>{w, rem_dim},
				0,
				d);
	}
};

template <typename T, Layout L>
LDLT_NO_INLINE void
row_delete_single(LdltViewMut<T, L> out_l, LdltView<T, L> in_l, i32 i) {
	i32 dim = in_l.d.dim;

	bool inplace = out_l.l.data == in_l.l.data;

	// top left
	if (!inplace) {
		to_eigen_matrix_mut(MatrixViewMut<T, L>{
				out_l.l.data,
				i,
				i,
				out_l.l.outer_stride,
		}) = to_eigen_matrix(MatrixView<T, L>{
				in_l.l.data,
				i,
				i,
				in_l.l.outer_stride,
		});
		std::copy(in_l.d.data, in_l.d.data + i, out_l.d.data);
	}
	if ((i + 1) == dim) {
		return;
	}

	// bottom left
	RowDeleteImpl<L>::copy_block(
			out_l.l.block(i, 0, 0, 0).data,
			in_l.l.block(i + 1, 0, 0, 0).data,
			dim - i,
			i,
			out_l.l.outer_stride,
			in_l.l.outer_stride);

	i32 rem_dim = dim - i - 1;
	RowDeleteImpl<L>::handle_bottom_right(
			LdltViewMut<T, L>{
					out_l.l.block(i, i, rem_dim, rem_dim),
					out_l.d.segment(i, rem_dim),
			},
			LdltView<T, L>{
					in_l.l.block(i + 1, i + 1, rem_dim, rem_dim),
					in_l.d.segment(i + 1, rem_dim),
			},
			in_l.l.block(i + 1, i, 0, 0).data,
			in_l.d(i),
			rem_dim,
			inplace);
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
	void operator()(
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
struct diagonal_update {
	template <
			typename Scalar,
			Layout L,
			typename S = diagonal_update_strategies::MultiPass>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, L> out,
			LdltView<Scalar, L> in,
			VectorView<Scalar> diag_diff,
			i32 start_index,
			S /*strategy_tag*/ = S{}) const {
		detail::DiagonalUpdateImpl<S>::fn(out, in, diag_diff, start_index);
	}
};

// output.dim == input.dim + 1
struct row_append {
	template <typename Scalar, Layout L>
	LDLT_INLINE void operator()(
			LdltViewMut<Scalar, L> out,
			LdltView<Scalar, L> in,
			VectorView<Scalar> new_row) const {
		detail::row_append(out, in, new_row);
	}
};

// output.dim == input.dim - 1
struct row_delete {
	template <typename Scalar, Layout L>
	LDLT_INLINE void operator()( //
			LdltViewMut<Scalar, L> out,
			LdltView<Scalar, L> in,
			i32 row_idx) const {
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
