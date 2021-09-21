#include <ldlt/factorize.hpp>

namespace ldlt {
namespace detail {
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f32>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f32>);
LDLT_EXPLICIT_TPL_DEF(1, factorize_unblocked<f64>);
LDLT_EXPLICIT_TPL_DEF(2, factorize_blocked<f64>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f32>);
LDLT_EXPLICIT_TPL_DEF(3, compute_permutation<f64>);
} // namespace detail
} // namespace ldlt
