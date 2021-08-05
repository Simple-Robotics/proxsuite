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

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
