#ifndef LDLT_MACROS_HPP_TSAOHJEXS
#define LDLT_MACROS_HPP_TSAOHJEXS

#include "ldlt/detail/hedley.h"

#define __VEG_PP_CALL(M, ...) M(__VA_ARGS__)
#define __VEG_PP_CALL2(M, ...) M(__VA_ARGS__)

#define LDLT_PP_2ND(...) __VEG_PP_HEAD(__VEG_PP_TAIL(__VA_ARGS__))
#define LDLT_PP_3RD(...)                                                       \
	__VEG_PP_HEAD(__VEG_PP_TAIL(__VEG_PP_TAIL(__VA_ARGS__)))
#define LDLT_PP_4TH(...)                                                       \
	__VEG_PP_HEAD(__VEG_PP_TAIL(__VEG_PP_TAIL(__VEG_PP_TAIL(__VA_ARGS__))))

#define LDLT_NOM_SEMICOLON static_assert(true, ".")

#define LDLT_PP_IMPL_IS_1(_0, _1, _2, _3, _4, _5, _7, N, ...) N
#define LDLT_PP_IS_SINGLE_ARG(...)                                             \
	LDLT_PP_IMPL_IS_1(__VA_ARGS__, 0, 0, 0, 0, 0, 0, 1, 1)

#define LDLT_PP_HEAD_ROBUST_0(x, ...) x
#define LDLT_PP_HEAD_ROBUST_1(...) __VA_ARGS__

#define LDLT_PP_TAIL_ROBUST_0(x, ...) __VA_ARGS__
#define LDLT_PP_TAIL_ROBUST_1(...)

#define LDLT_PP_HEAD_ROBUST(...)                                               \
	__VEG_PP_CAT2(LDLT_PP_HEAD_ROBUST_, LDLT_PP_IS_SINGLE_ARG(__VA_ARGS__))      \
	(__VA_ARGS__)

#define LDLT_PP_TAIL_ROBUST(...)                                               \
	__VEG_PP_CAT2(LDLT_PP_TAIL_ROBUST_, LDLT_PP_IS_SINGLE_ARG(__VA_ARGS__))      \
	(__VA_ARGS__)

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

#ifndef LDLT_MAX_STACK_ALLOC_SIZE
#define LDLT_MAX_STACK_ALLOC_SIZE (1024ULL * 8ULL) /* 8KiB */
#endif

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
#define LDLT_FREEA(x) ((void)x)
#else
#define LDLT_HAS_ALLOCA 0
#endif

#endif

#define LDLT_CACHELINE_BYTES 64

#define LDLT_IMPL_TO_ARGS_DISPATCH_Vec(Dim) Dim
#define LDLT_IMPL_TO_ARGS_DISPATCH_Mat(Rows, Cols) Rows, Cols

#define LDLT_IMPL_TO_ARGS(Dim) __VEG_PP_CAT(LDLT_IMPL_TO_ARGS_DISPATCH_, Dim)

#define LDLT_IMPL_ALIGN(I, _, NameTagDimAlignType)                             \
	::ldlt::usize{(LDLT_PP_4TH NameTagDimAlignType)},

#define LDLT_IMPL_LOCAL_DIMS_VEC2(I, Dim, Align, Type)                         \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(dim, I)) = ::ldlt::usize(Dim);      \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(size_bytes, I)) =                   \
			::ldlt::detail::uround_up(                                               \
					sizeof(Type) * LDLT_ID(__VEG_PP_CAT(dim, I)),                        \
					::ldlt::usize{(Align)});

#define LDLT_IMPL_LOCAL_DIMS_MAT2(I, Rows, Cols, Align, Type)                  \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(rows, I)) = ::ldlt::usize(Rows);    \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(cols, I)) = ::ldlt::usize(Cols);    \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(stride_bytes, I)) =                 \
			::ldlt::detail::uround_up(                                               \
					sizeof(Type) * LDLT_ID(__VEG_PP_CAT(rows, I)),                       \
					::ldlt::usize{(Align)});                                             \
	::ldlt::usize const LDLT_ID(__VEG_PP_CAT(size_bytes, I)) =                   \
			LDLT_ID(__VEG_PP_CAT(cols, I)) * LDLT_ID(__VEG_PP_CAT(stride_bytes, I));

#define LDLT_IMPL_LOCAL_DIMS_VEC(I, Name, Tag, Dim, Align, Type)               \
	LDLT_IMPL_LOCAL_DIMS_VEC2(I, LDLT_IMPL_TO_ARGS(Dim), Align, Type)

#define LDLT_IMPL_LOCAL_DIMS_MAT(I, Name, Tag, Dim, Align, Type)               \
	__VEG_PP_CALL2(                                                              \
			LDLT_IMPL_LOCAL_DIMS_MAT2, I, LDLT_IMPL_TO_ARGS(Dim), Align, Type)

#define LDLT_IMPL_LOCAL_DIMS_DISPATCH_Vec(Dim) LDLT_IMPL_LOCAL_DIMS_VEC
#define LDLT_IMPL_LOCAL_DIMS_DISPATCH_Mat(Rows, Cols) LDLT_IMPL_LOCAL_DIMS_MAT

#define LDLT_IMPL_MIN_ALIGN_DISPATCH_Vec(Dim) LDLT_IMPL_MIN_ALIGN_VEC
#define LDLT_IMPL_MIN_ALIGN_DISPATCH_Mat(Rows, Cols) LDLT_IMPL_MIN_ALIGN_MAT

#define LDLT_IMPL_LOCAL_DIMS(I, _, NameTagDimAlignType)                        \
	__VEG_PP_CALL(                                                               \
			__VEG_PP_CAT(                                                            \
					LDLT_IMPL_LOCAL_DIMS_DISPATCH_, LDLT_PP_3RD NameTagDimAlignType),    \
			I,                                                                       \
			__VEG_PP_REMOVE_PAREN NameTagDimAlignType)                               \
	LDLT_ID(total_size_bytes) +=                                                 \
			LDLT_ID(__VEG_PP_CAT(size_bytes, I)) +                                   \
			(::ldlt::usize{LDLT_PP_4TH NameTagDimAlignType} - LDLT_ID(min_align));

#define LDLT_IMPL_MAKE_WORKSPACE2_Vec(I, Name, Align, Type)                    \
	::ldlt::VectorViewMut<Type> Name = {                                         \
			::ldlt::tags::FromPtrSize{},                                             \
			LDLT_ID(__VEG_PP_CAT(array_manager, I))._.data,                          \
			::ldlt::isize(LDLT_ID(__VEG_PP_CAT(dim, I))),                            \
	}

#define LDLT_IMPL_MAKE_WORKSPACE2_Mat(I, Name, Align, Type)                    \
	static_assert(                                                               \
			((sizeof(Type) % ::ldlt::usize{(Align)} == 0) ||                         \
	     (::ldlt::usize{(Align)} % sizeof(Type) == 0)),                          \
			".");                                                                    \
	::ldlt::MatrixViewMut<Type, ::ldlt::Layout::colmajor> Name = {               \
			::ldlt::tags::FromPtrRowsColsStride{},                                   \
			LDLT_ID(__VEG_PP_CAT(array_manager, I))._.data,                          \
			::ldlt::isize(LDLT_ID(__VEG_PP_CAT(rows, I))),                           \
			::ldlt::isize(LDLT_ID(__VEG_PP_CAT(cols, I))),                           \
			::ldlt::isize(LDLT_ID(__VEG_PP_CAT(stride_bytes, I)) / sizeof(Type)),    \
	}

#define LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_Vec(Dim)                            \
	LDLT_IMPL_MAKE_WORKSPACE2_Vec

#define LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_Mat(Rows, Cols)                     \
	LDLT_IMPL_MAKE_WORKSPACE2_Mat

#define LDLT_IMPL_MAKE_WORKSPACE2(I, Name, Tag, Dim, Align, Type)              \
	::ldlt::detail::ManagedArray<Type> LDLT_ID(__VEG_PP_CAT(array_manager, I)){  \
			::ldlt::detail::malloca_tags::Tag{},                                     \
			typename ::ldlt::detail::ManagedArray<Type>::Inner{                      \
					static_cast<Type*>(::ldlt::detail::next_aligned(                     \
							LDLT_ID(current_ptr), ::ldlt::usize{(Align)})),                  \
					(LDLT_ID(__VEG_PP_CAT(size_bytes, I)) / sizeof(Type)),               \
			},                                                                       \
	};                                                                           \
	__VEG_PP_CAT(LDLT_IMPL_MAKE_WORKSPACE2_DISPATCH_, Dim)                       \
	(I, Name, Align, Type);                                                      \
	LDLT_ID(current_ptr) += LDLT_ID(__VEG_PP_CAT(size_bytes, I));

#define LDLT_IMPL_MAKE_WORKSPACE(I, _, NameTagDimAlignType)                    \
	__VEG_PP_CALL(                                                               \
			LDLT_IMPL_MAKE_WORKSPACE2, I, __VEG_PP_REMOVE_PAREN NameTagDimAlignType)

#define LDLT_WORKSPACE_MEMORY(Name, Tag, Dim, Align, Type)                     \
	LDLT_IMPL_MULTI_WORKSPACE_MEMORY(((Name, Tag, Dim, Align, Type)))

#define LDLT_MULTI_WORKSPACE_MEMORY(...)                                       \
	LDLT_IMPL_MULTI_WORKSPACE_MEMORY((__VA_ARGS__))

#define LDLT_IMPL_MULTI_WORKSPACE_MEMORY(NamesTagsDimsAlignsTypes)             \
	constexpr ::ldlt::usize LDLT_ID(min_align) =                                 \
			::ldlt::detail::cx_min_list({__VEG_PP_TUPLE_FOR_EACH_I(                  \
					LDLT_IMPL_ALIGN, _, NamesTagsDimsAlignsTypes)});                     \
	::ldlt::usize LDLT_ID(total_size_bytes) = 0;                                 \
	__VEG_PP_TUPLE_FOR_EACH_I(LDLT_IMPL_LOCAL_DIMS, _, NamesTagsDimsAlignsTypes) \
	LDLT_IMPL_GET_MALLOCA(                                                       \
			LDLT_ID(w_malloca), LDLT_ID(total_size_bytes), LDLT_ID(min_align));      \
	unsigned char* LDLT_ID(current_ptr) =                                        \
			static_cast<unsigned char*>(LDLT_ID(w_malloca)._.malloca_ptr);           \
	__VEG_PP_TUPLE_FOR_EACH_I(                                                   \
			LDLT_IMPL_MAKE_WORKSPACE, _, NamesTagsDimsAlignsTypes)                   \
	LDLT_NOM_SEMICOLON

#if LDLT_HAS_ALLOCA

#define LDLT_IMPL_GET_MALLOCA(Name, SizeBytes, Align)                          \
	static_assert(                                                               \
			::ldlt::usize(Align) > 0, "Align must be a constant expression");        \
	::ldlt::usize const LDLT_ID(malloca_size_bytes) =                            \
			::ldlt::usize(SizeBytes) + ::ldlt::usize{Align};                         \
	::ldlt::detail::ManagedMalloca const Name {                                  \
		::ldlt::detail::malloca_tags::Uninit{},                                    \
				::ldlt::detail::ManagedMalloca::Inner{                                 \
						((LDLT_ID(malloca_size_bytes)) <                                   \
		         ::ldlt::usize{LDLT_MAX_STACK_ALLOC_SIZE})                         \
								? (LDLT_ALLOCA(LDLT_ID(malloca_size_bytes)))                   \
								: (::std::malloc(LDLT_ID(malloca_size_bytes))),                \
						LDLT_ID(malloca_size_bytes),                                       \
				},                                                                     \
	}

#else

#define LDLT_IMPL_GET_MALLOCA(Name, SizeBytes, Align)                          \
	alignas(::ldlt::usize{Align}) unsigned char LDLT_ID(                         \
			buf)[LDLT_MAX_STACK_ALLOC_SIZE];                                         \
	::ldlt::usize const LDLT_ID(malloca_size_bytes) =                            \
			::ldlt::usize(SizeBytes) + ::ldlt::usize{Align};                         \
	::ldlt::detail::ManagedMalloca Name {                                        \
		::ldlt::detail::malloca_tags::Uninit{},                                    \
				::ldlt::detail::ManagedMalloca::Inner{                                 \
						(((SizeBytes)) < ::ldlt::usize{LDLT_MAX_STACK_ALLOC_SIZE})         \
								? static_cast<void*>(LDLT_ID(buf))                             \
								: (::std::malloc(LDLT_ID(malloca_size_bytes))),                \
						(SizeBytes),                                                       \
				},                                                                     \
	}

#endif

#ifdef __clang__
#define LDLT_MINSIZE __attribute__((minsize))
#else
#define LDLT_MINSIZE
#endif

#define LDLT_CONCEPT(...) VEG_CONCEPT_MACRO(::ldlt::concepts, __VA_ARGS__)
#define LDLT_CHECK_CONCEPT(...)                                                \
	VEG_CHECK_CONCEPT_MACRO(::ldlt::concepts, __VA_ARGS__)

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
