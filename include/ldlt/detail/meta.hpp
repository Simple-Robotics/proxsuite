#ifndef INRIA_LDLT_META_HPP_VHFXDOQHS
#define INRIA_LDLT_META_HPP_VHFXDOQHS

#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <atomic>
#include "ldlt/detail/macros.hpp"

namespace ldlt {
using usize = decltype(sizeof(0));
namespace detail {
namespace meta_ {

template <typename T, T... Nums>
struct integer_sequence;

#if defined(__has_builtin)
#define LDLT_HAS_BUILTIN(x) __has_builtin(x)
#else
#define LDLT_HAS_BUILTIN(x) 0
#endif

#if LDLT_HAS_BUILTIN(__make_integer_seq)

template <typename T, T N>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, N>;

#elif __GNUC__ >= 8

template <typename T, T N>
using make_integer_sequence = integer_sequence<T, __integer_pack(N)...>;

#else

namespace internal {

template <typename Seq1, typename Seq2>
struct _merge;

template <typename Seq1, typename Seq2>
struct _merge_p1;

template <typename T, T... Nums1, T... Nums2>
struct _merge<integer_sequence<T, Nums1...>, integer_sequence<T, Nums2...>> {
	using type = integer_sequence<T, Nums1..., (sizeof...(Nums1) + Nums2)...>;
};

template <typename T, T... Nums1, T... Nums2>
struct _merge_p1<integer_sequence<T, Nums1...>, integer_sequence<T, Nums2...>> {
	using type = integer_sequence<
			T,
			Nums1...,
			(sizeof...(Nums1) + Nums2)...,
			sizeof...(Nums1) + sizeof...(Nums2)>;
};

template <typename T, usize N, bool Even = (N % 2) == 0>
struct _make_integer_sequence {
	using type = typename _merge<
			typename _make_integer_sequence<T, N / 2>::type,
			typename _make_integer_sequence<T, N / 2>::type>::type;
};

template <typename T, usize N>
struct _make_integer_sequence<T, N, false> {
	using type = typename _merge_p1<
			typename _make_integer_sequence<T, N / 2>::type,
			typename _make_integer_sequence<T, N / 2>::type>::type;
};

template <typename T>
struct _make_integer_sequence<T, 0> {
	using type = integer_sequence<T>;
};
template <typename T>
struct _make_integer_sequence<T, 1> {
	using type = integer_sequence<T, 0>;
};

} // namespace internal

template <typename T, T N>
using make_integer_sequence =
		typename internal::_make_integer_sequence<T, N>::type;

#endif

template <usize N>
using make_index_sequence = make_integer_sequence<usize, N>;

template <typename... Ts>
struct type_sequence;

} // namespace meta_

template <typename T, T N>
using make_integer_sequence = meta_::make_integer_sequence<T, N>*;
template <usize N>
using make_index_sequence = meta_::make_integer_sequence<usize, N>*;

template <typename T, T... Nums>
using integer_sequence = meta_::integer_sequence<T, Nums...>*;
template <usize... Nums>
using index_sequence = integer_sequence<usize, Nums...>;

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

template <typename Fn>
struct FnInfo;
template <typename Ret_, typename... Args>
struct FnInfo<auto(Args...)->Ret_> {
	template <usize I>
	using Arg = ith<I, Args...>;
	using Ret = Ret_;
};

template <typename Char, Char... Cs>
struct LiteralConstant {};

template <typename LiteralType, typename Seq>
struct ExtractCharsImpl;

template <typename LiteralType, usize... Is>
struct ExtractCharsImpl<LiteralType, index_sequence<Is...>> {
	using Type =
			LiteralConstant<typename LiteralType::Type, LiteralType::value()[Is]...>;
};

template <typename LiteralType>
auto ExtractChars(LiteralType /*unused*/) -> typename ExtractCharsImpl<
		LiteralType,
		make_index_sequence<LiteralType::size()>>::Type {
	return {};
}

using Clock = std::chrono::steady_clock;
using Time = typename Clock::time_point;
using Duration = typename Time::duration;
using Container = std::vector<Duration>;
struct ContainerRefMut {
	Container& ref;
};
using InnerMap = std::map<std::string, ContainerRefMut>;
using OuterMap = std::map<std::string, InnerMap>;

template <typename T>
struct UniqueObserver {
	T* ptr;
	LDLT_INLINE explicit UniqueObserver(T* ptr_) noexcept : ptr{ptr_} {}

	UniqueObserver(UniqueObserver const& rhs) noexcept = delete;
	auto operator=(UniqueObserver const& rhs) noexcept
			-> UniqueObserver& = delete;

	LDLT_INLINE UniqueObserver(UniqueObserver&& rhs) noexcept : ptr(rhs.ptr) {
		rhs.ptr = {};
	}
	LDLT_INLINE auto operator=(UniqueObserver&& rhs) noexcept -> UniqueObserver& {
		T* tmp = rhs.ptr;
		rhs.ptr = {};
		ptr = rhs.ptr;
	}
	~UniqueObserver() = default;
};

struct ScopedTimer {
	Container* timings;
	Time begin;

	LDLT_INLINE ~ScopedTimer() {
		if (timings != nullptr) {
			destroy();
		}
	}

	LDLT_NO_INLINE void destroy() const {
		timings->push_back(Clock::now() - begin);
	}

	ScopedTimer(ScopedTimer const&) = delete;
	auto operator=(ScopedTimer const&) -> ScopedTimer& = delete;

	ScopedTimer(ScopedTimer&&) = default;
	auto operator=(ScopedTimer&&) -> ScopedTimer& = default;

	explicit ScopedTimer(Container* ref) noexcept : timings{ref}, begin{} {
		if (timings != nullptr) {
			begin = Clock::now();
		}
	}
};

LDLT_NO_INLINE inline auto container_init_0() -> Container {
	Container vec;
	vec.reserve(4096);
	return vec;
}

template <typename... Args>
struct SectionTimingMap {
	static auto ref() -> OuterMap& {
		static OuterMap v;
		return v;
	}
	static void register_container(
			std::string outer_name, std::string inner_name, Container& c) {
		ref()[LDLT_FWD(outer_name)].insert({
				LDLT_FWD(inner_name),
				{c},
		});
	}
};

inline auto benchmark_var() -> std::atomic<bool>& {
	static std::atomic<bool> inner(true);
	return inner;
}

template <typename CharO, CharO... COs>
struct SectionTimingOuterTag {

	template <typename CharI, CharI... CIs>
	struct SectionTimingInnerTag {

		template <typename... Args>
		struct Type {

			LDLT_NO_INLINE static auto container_init() -> Container& {
				static auto vec = container_init_0();
				CharO outer[] = {COs...};
				CharI inner[] = {CIs...};
				SectionTimingMap<Args...>::register_container(
						std::string(outer, outer + sizeof(outer) / sizeof(outer[0]) - 1),
						std::string(inner, inner + sizeof(inner) / sizeof(inner[0]) - 1),
						vec);
				return vec;
			}

			static auto ref() -> Container& {
				static auto& v = container_init();
				return v;
			}
			auto scoped() noexcept -> ScopedTimer {
				return ScopedTimer{
						benchmark_var().load(std::memory_order_acquire) //
								? std::addressof(ref())
								: nullptr};
			}
		};
	};
};

template <
		typename... Args,
		typename CharOuter,
		CharOuter... COs,
		typename CharInner,
		CharInner... CIs>
auto section_timings(
		LiteralConstant<CharOuter, COs...> /*unused*/,
		LiteralConstant<CharInner, CIs...> /*unused*/) noexcept ->
		typename SectionTimingOuterTag<CharOuter, COs...>::
				template SectionTimingInnerTag<CharInner, CIs...>::template Type<
						Args...> {
	return {};
}

inline void toggle_benchmarks(bool enable) {
	benchmark_var().store(enable, std::memory_order_release);
}
} // namespace detail
} // namespace ldlt

#define LDLT_LITERAL_TO_CONSTANT(Literal)                                      \
	(::ldlt::detail::ExtractChars([]() /* NOLINT */ noexcept {                   \
		struct LDLT_HiddenType {                                                   \
			static constexpr auto value() noexcept -> decltype(Literal) {            \
				return Literal;                                                        \
			};                                                                       \
			using Type = typename std::remove_const<                                 \
					typename std::remove_reference<decltype(value()[0])>::type>::type;   \
			static constexpr auto size() noexcept -> ::ldlt::usize {                 \
				return sizeof(value()) / sizeof(Type);                                 \
			}                                                                        \
		};                                                                         \
		return LDLT_HiddenType{};                                                  \
	}()))

#define LDLT_IMPL_TIMINGS(OuterTag, ...)                                       \
	(::ldlt::detail::section_timings<LDLT_PP_TAIL_ROBUST(__VA_ARGS__)>(          \
			LDLT_LITERAL_TO_CONSTANT(OuterTag),                                      \
			LDLT_LITERAL_TO_CONSTANT(LDLT_PP_HEAD_ROBUST(__VA_ARGS__))))

#define LDLT_SCOPE_TIMER(...) (LDLT_IMPL_TIMINGS(__VA_ARGS__).scoped())

#define LDLT_DECL_SCOPE_TIMER(...)                                             \
	auto&& LDLT_PP_CAT2(_ldlt_dummy_timer_var_, __LINE__) =                      \
			LDLT_SCOPE_TIMER(__VA_ARGS__);                                           \
	((void)LDLT_PP_CAT2(_ldlt_dummy_timer_var_, __LINE__))

#define LDLT_GET_DURATIONS(...) (LDLT_IMPL_TIMINGS(__VA_ARGS__).ref())
#define LDLT_GET_MAP(...) (::ldlt::detail::SectionTimingMap<__VA_ARGS__>::ref())

#define LDLT_IMPL_GET_PARAM(Fn, Idx)                                           \
	typename ::ldlt::detail::FnInfo<decltype Fn>::template Arg<(Idx)>,

#define LDLT_IMPL_GET_PARAMS_0(NParams, ...)                                   \
	__VEG_PP_TUPLE_FOR_EACH(                                                     \
			LDLT_IMPL_GET_PARAM,                                                     \
			(__VA_ARGS__),                                                           \
			__VEG_PP_MAKE_TUPLE(__VEG_IMPL_PP_DEC(NParams)))

#define LDLT_IMPL_GET_PARAMS_1(NParams, ...)

#define LDLT_IMPL_GET_PARAMS(NParams, ...)                                     \
	__VEG_PP_CAT2(LDLT_IMPL_GET_PARAMS_, __VEG_IMPL_PP_IS_1(NParams))            \
	(NParams, __VA_ARGS__)

#define LDLT_EXPLICIT_TPL_DEF(NParams, ...)                                    \
	template auto __VA_ARGS__(                                                   \
			LDLT_IMPL_GET_PARAMS(NParams, __VA_ARGS__)                               \
					typename ::ldlt::detail::FnInfo<                                     \
							decltype(__VA_ARGS__)>::template Arg<(NParams)-1>)               \
			->typename ::ldlt::detail::FnInfo<decltype(__VA_ARGS__)>::Ret
#define LDLT_EXPLICIT_TPL_DECL(NParams, ...)                                   \
	extern LDLT_EXPLICIT_TPL_DEF(NParams, __VA_ARGS__)

#endif /* end of include guard INRIA_LDLT_META_HPP_VHFXDOQHS */
