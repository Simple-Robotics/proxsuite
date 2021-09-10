#include <ldlt/update.hpp>

namespace ldlt {
namespace detail {
template void rank1_update(
		LdltViewMut<f32>, LdltView<f32>, VectorViewMut<f32>, isize, f32);
template void rank1_update(
		LdltViewMut<f64>, LdltView<f64>, VectorViewMut<f64>, isize, f64);
} // namespace detail
} // namespace ldlt
