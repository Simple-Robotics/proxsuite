#include <ldlt/update.hpp>

template void ldlt::detail::rank1_update(
		LdltViewMut<f32, colmajor>,
		LdltView<f32, colmajor>,
		VectorViewMut<f32>,
		i32,
		f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f32, rowmajor>,
		LdltView<f32, rowmajor>,
		VectorViewMut<f32>,
		i32,
		f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, colmajor>,
		LdltView<f64, colmajor>,
		VectorViewMut<f64>,
		i32,
		f64);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, rowmajor>,
		LdltView<f64, rowmajor>,
		VectorViewMut<f64>,
		i32,
		f64);
