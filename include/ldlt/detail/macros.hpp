#ifndef LDLT_MACROS_HPP_TSAOHJEXS
#define LDLT_MACROS_HPP_TSAOHJEXS

#include "ldlt/detail/hedley.h"

#define LDLT_REMOVE_PAREN(...) __VA_ARGS__

#define LDLT_NOM_SEMICOLON static_assert(true, ".")

#ifndef LDLT_INLINE
#define LDLT_INLINE HEDLEY_ALWAYS_INLINE
#endif

#ifndef LDLT_NO_INLINE
#define LDLT_NO_INLINE HEDLEY_NEVER_INLINE
#endif

#define LDLT_FWD(x) (static_cast<decltype(x)&&>(x))
#define LDLT_DECLVAL(...) (static_cast<auto (*)()->__VA_ARGS__>(nullptr)())

#ifdef __clang__
#define LDLT_FP_PRAGMA _Pragma("STDC FP_CONTRACT ON")
#else
#define LDLT_FP_PRAGMA
#endif

#define LDLT_MAX_STACK_ALLOC_SIZE (1024U * 8U) /* 8KiB */

#ifndef LDLT_HAS_ALLOCA

#if defined(HEDLEY_MSVC_VERSION)
#include <malloc.h>

#define LDLT_HAS_ALLOCA 1
#define LDLT_ALLOCA(x) (_malloca(x))
#define LDLT_FREEA(x) (_freea(x))
#elif defined(HEDLEY_GCC_VERSION) || defined(__clang__)
#include <alloca.h>

#define LDLT_HAS_ALLOCA 1
#define LDLT_ALLOCA(x) (alloca(x))
#define LDLT_FREEA(x) ((void)0)
#else
#define LDLT_HAS_ALLOCA 0
#endif

#endif

#if LDLT_HAS_ALLOCA

#define LDLT_WORKSPACE_MEMORY(Name, Count, Type)                               \
	usize const _alloc##__LINE__ =                                               \
			usize(Count) * sizeof(Type) + UniqueMalloca<Type>::align;                \
	UniqueMalloca<Type> _malloca_handler##__LINE__{                              \
			UniqueMalloca<Type>::can_alloca(usize(Count))                            \
					? (LDLT_ALLOCA(_alloc##__LINE__))                                    \
					: ::std::malloc(_alloc##__LINE__),                                   \
			usize(Count),                                                            \
	};                                                                           \
	Type* Name = _malloca_handler##__LINE__.data;                                \
	static_assert(true, ".")

#else

#define LDLT_WORKSPACE_MEMORY(Name, Count, Type)                               \
	alignas(UniqueMalloca<Type>::align) unsigned char                            \
			_buf##__LINE__[LDLT_MAX_STACK_ALLOC_SIZE];                               \
	usize const _alloc##__LINE__ =                                               \
			usize(Count) * sizeof(Type) + UniqueMalloca<Type>::align;                \
	UniqueMalloca<Type> _malloca_handler##__LINE__{                              \
			UniqueMalloca<Type>::can_alloca(usize(Count))                            \
					? static_cast<void*>(_buf)                                           \
					: ::std::malloc(_alloc##__LINE__),                                   \
			usize(Count),                                                            \
	};                                                                           \
	Type* Name = _malloca_handler##__LINE__.data;                                \
	static_assert(true, ".")

#endif

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
