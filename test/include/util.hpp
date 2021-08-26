#ifndef INRIA_LDLT_UTIL_HPP_ZX9HNY5GS
#define INRIA_LDLT_UTIL_HPP_ZX9HNY5GS

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <utility>
#include <ldlt/views.hpp>
#include <ldlt/qp/views.hpp>
#include <ldlt/detail/meta.hpp>

template <typename T, ldlt::Layout L>
using Mat = Eigen::Matrix<
		T,
		Eigen::Dynamic,
		Eigen::Dynamic,
		(L == ldlt::colmajor) ? Eigen::ColMajor : Eigen::RowMajor>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <
		typename MatLhs,
		typename MatRhs,
		typename T = typename MatLhs::Scalar>
auto matmul(MatLhs const& a, MatRhs const& b) -> Mat<T, ldlt::colmajor> {
	using Upscaled = typename std::
			conditional<std::is_floating_point<T>::value, long double, T>::type;

	return (Mat<T, ldlt::colmajor>(a).template cast<Upscaled>().operator*(
							Mat<T, ldlt::colmajor>(b).template cast<Upscaled>()))
	    .template cast<T>();
}

template <
		typename MatLhs,
		typename MatMid,
		typename MatRhs,
		typename T = typename MatLhs::Scalar>
auto matmul3(MatLhs const& a, MatMid const& b, MatRhs const& c)
		-> Mat<T, ldlt::colmajor> {
	return ::matmul(::matmul(a, b), c);
}

namespace ldlt {
namespace detail {
template <typename... Ts>
using type_sequence = meta_::type_sequence<Ts...>*;

template <usize I, typename T>
struct HollowLeaf {};
template <typename ISeq, typename... Ts>
struct HollowIndexedTuple;
template <usize... Is, typename... Ts>
struct HollowIndexedTuple<index_sequence<Is...>, Ts...>
		: HollowLeaf<Is, Ts>... {};

template <usize I, typename T>
auto get_type(HollowLeaf<I, T> const*) noexcept -> T;

template <usize I>
struct pack_ith_elem {
	template <typename... Ts>
	using Type = decltype(detail::get_type<I>(
			static_cast<
					HollowIndexedTuple<make_index_sequence<sizeof...(Ts)>, Ts...>*>(
					nullptr)));
};

#if LDLT_HAS_BUILTIN(__type_pack_element)
template <usize I, typename... Ts>
using ith = __type_pack_element<I, Ts...>;
#else
template <usize I, typename... Ts>
using ith = typename pack_ith_elem<I>::template Type<Ts...>;
#endif

template <typename T>
struct type_list_ith_elem_impl;

template <usize I, typename List>
using typeseq_ith =
		typename type_list_ith_elem_impl<List>::template ith_elem<I>;

template <typename... Ts>
struct type_list_ith_elem_impl<type_sequence<Ts...>> {
	template <usize I>
	using ith_elem = ith<I, Ts...>;
};
template <typename T, T Val>
struct constant {
	static constexpr T value = Val;
};
} // namespace detail
} // namespace ldlt

LDLT_DEFINE_TAG(random_with_dim_and_n_eq, RandomWithDimAndNeq);
LDLT_DEFINE_TAG(from_data, FromData);
template <typename Scalar>
struct Qp {
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> H;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> g;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> b;

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> solution;

	Qp(FromData /*tag*/,
	   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> H_,
	   Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> g_,
	   Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A_,
	   Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> b_) noexcept
			: H(LDLT_FWD(H_)),
				g(LDLT_FWD(g_)),
				A(LDLT_FWD(A_)),
				b(LDLT_FWD(b_)),
				solution(H.rows() + A.rows()) {

		ldlt::i32 dim = ldlt::i32(H.rows());
		ldlt::i32 n_eq = ldlt::i32(A.rows());

		auto kkt_mat =
				Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(
						dim + n_eq, dim + n_eq);

		kkt_mat.topLeftCorner(dim, dim) = H;
		kkt_mat.topRightCorner(dim, n_eq) = A.transpose();
		kkt_mat.bottomLeftCorner(n_eq, dim) = A;
		kkt_mat.bottomRightCorner(n_eq, n_eq).setZero();

		solution.topRows(dim) = -g;
		solution.bottomRows(n_eq) = b;
		kkt_mat.ldlt().solveInPlace(solution);
	}

	Qp(RandomWithDimAndNeq /*tag*/, ldlt::i32 dim, ldlt::i32 n_eq)
			: H(dim, dim), g(dim), A(n_eq, dim), b(n_eq), solution(dim + n_eq) {
		A.setRandom();

		// 1/2 (x-sol)T H (x-sol)
		// 1/2 xT H x - (H sol).T x
		solution.setRandom();
		auto primal_solution = solution.topRows(dim);
		auto dual_solution = solution.bottomRows(n_eq);

		{
			H.setRandom();
			H = H * H.transpose();
			H.diagonal().array() += Scalar(1e-3);
		}

		g = -H * primal_solution - A.transpose() * dual_solution;
		b = A * primal_solution;
	}

	auto as_view() -> qp::QpView<Scalar, ldlt::colmajor, ldlt::colmajor> {
		return {
				ldlt::detail::from_eigen_matrix(H),
				ldlt::detail::from_eigen_vector(g),
				ldlt::detail::from_eigen_matrix(A),
				ldlt::detail::from_eigen_vector(b),
				{},
				{},
		};
	}
	auto as_mut() -> qp::QpViewMut<Scalar, ldlt::colmajor, ldlt::colmajor> {
		return {
				ldlt::detail::from_eigen_matrix_mut(H),
				ldlt::detail::from_eigen_vector_mut(g),
				ldlt::detail::from_eigen_matrix_mut(A),
				ldlt::detail::from_eigen_vector_mut(b),
				{nullptr, 0, ldlt::i32(H.rows()), 0},
				{nullptr, 0},
		};
	}
};

struct EigenNoAlloc {
	EigenNoAlloc(EigenNoAlloc&&) = delete;
	EigenNoAlloc(EigenNoAlloc const&) = delete;
	auto operator=(EigenNoAlloc&&) -> EigenNoAlloc& = delete;
	auto operator=(EigenNoAlloc const&) -> EigenNoAlloc& = delete;

	EigenNoAlloc() noexcept {
#if defined(EIGEN_RUNTIME_NO_MALLOC)
		Eigen::internal::set_is_malloc_allowed(false);
#endif
	}
	~EigenNoAlloc() noexcept {
#if defined(EIGEN_RUNTIME_NO_MALLOC)
		Eigen::internal::set_is_malloc_allowed(true);
#endif
	}
};

#endif /* end of include guard INRIA_LDLT_UTIL_HPP_ZX9HNY5GS */
