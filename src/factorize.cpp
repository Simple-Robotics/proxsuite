#include <ldlt/factorize.hpp>

template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, colmajor>, MatrixView<f32, colmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, rowmajor>, MatrixView<f32, colmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, colmajor>, MatrixView<f32, rowmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f32, rowmajor>, MatrixView<f32, rowmajor>);

template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, colmajor>, MatrixView<f64, colmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, rowmajor>, MatrixView<f64, colmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, colmajor>, MatrixView<f64, rowmajor>);
template void ldlt::detail::factorize_ldlt_tpl(
		LdltViewMut<f64, rowmajor>, MatrixView<f64, rowmajor>);
