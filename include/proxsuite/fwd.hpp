//
// Copyright (c) 2022 INRIA
//

#ifndef __proxsuite_fwd_hpp__
#define __proxsuite_fwd_hpp__

#if __cplusplus >= 201703L
#define PROXSUITE_WITH_CPP_17
#endif
#if __cplusplus >= 201402L
#define PROXSUITE_WITH_CPP_14
#endif

#if defined(PROXSUITE_WITH_CPP_17)
#define PROXSUITE_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define PROXSUITE_MAYBE_UNUSED
#else
#define PROXSUITE_MAYBE_UNUSED __attribute__((__unused__))
#endif

#endif // #ifndef __proxsuite_fwd_hpp__
