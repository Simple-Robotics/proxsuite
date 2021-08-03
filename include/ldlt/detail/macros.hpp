#ifndef LDLT_MACROS_HPP_TSAOHJEXS
#define LDLT_MACROS_HPP_TSAOHJEXS

#include "ldlt/detail/hedley.h"

#ifndef LDLT_INLINE
#define LDLT_INLINE HEDLEY_ALWAYS_INLINE
#endif

#define LDLT_FWD(x) (static_cast<decltype(x)&&>(x))
#define LDLT_DECLVAL(...) (static_cast<auto (*)()->__VA_ARGS__>(nullptr)())

#endif /* end of include guard LDLT_MACROS_HPP_TSAOHJEXS */
