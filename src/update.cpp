#include <ldlt/update.hpp>

template void ldlt::detail::rank1_update(
		LdltViewMut<f32, colmajor>, LdltView<f32, colmajor>, VectorView<f32>, f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f32, rowmajor>, LdltView<f32, rowmajor>, VectorView<f32>, f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, colmajor>, LdltView<f64, colmajor>, VectorView<f64>, f64);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, rowmajor>, LdltView<f64, rowmajor>, VectorView<f64>, f64);
