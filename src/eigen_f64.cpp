#include <ldlt/views.hpp>

namespace ldlt {
namespace detail {
LDLT_EXPLICIT_TPL_DEF(4, noalias_mul_add<f64>);
LDLT_EXPLICIT_TPL_DEF(3, assign_cwise_prod<f64>);
LDLT_EXPLICIT_TPL_DEF(3, assign_scalar_prod<f64>);
LDLT_EXPLICIT_TPL_DEF(2, trans_tr_unit_up_solve_in_place_on_right<f64>);
LDLT_EXPLICIT_TPL_DEF(3, apply_diag_inv_on_right<f64>);
LDLT_EXPLICIT_TPL_DEF(3, apply_diag_on_right<f64>);
LDLT_EXPLICIT_TPL_DEF(3, noalias_mul_sub_tr_lo<f64>);
} // namespace detail
} // namespace ldlt
