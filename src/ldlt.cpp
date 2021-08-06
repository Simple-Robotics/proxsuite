#include <ldlt/solve.hpp>
#include <ldlt/factorize.hpp>
#include <ldlt/solve.hpp>

template void ldlt::detail::solve_impl(
		VectorViewMut<f32>, LdltView<f32, colmajor>, VectorView<f32>);
template void ldlt::detail::solve_impl(
		VectorViewMut<f64>, LdltView<f64, colmajor>, VectorView<f64>);
template void ldlt::detail::solve_impl(
		VectorViewMut<f32>, LdltView<f32, rowmajor>, VectorView<f32>);
template void ldlt::detail::solve_impl(
		VectorViewMut<f64>, LdltView<f64, rowmajor>, VectorView<f64>);

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

template void ldlt::detail::rank1_update(
		LdltViewMut<f32, colmajor>, LdltView<f32, colmajor>, VectorView<f32>, f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f32, rowmajor>, LdltView<f32, rowmajor>, VectorView<f32>, f32);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, colmajor>, LdltView<f64, colmajor>, VectorView<f64>, f64);
template void ldlt::detail::rank1_update(
		LdltViewMut<f64, rowmajor>, LdltView<f64, rowmajor>, VectorView<f64>, f64);
