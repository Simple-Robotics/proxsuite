#ifndef INRIA_LDLT_LDLT_HPP_VCVSK3EOS
#define INRIA_LDLT_LDLT_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"

namespace ldlt {

LDLT_DEFINE_TAG(decompose, Decompose);
LDLT_DEFINE_TAG(with_dim, WithDim);

template <typename T, Layout L>
struct Ldlt {
private:
	using LType = Eigen::Matrix<
			T,
			Eigen::Dynamic,
			Eigen::Dynamic,
			(L == colmajor ? Eigen::ColMajor : Eigen::RowMajor)>;
	using LTType = Eigen::Matrix<
			T,
			Eigen::Dynamic,
			Eigen::Dynamic,
			(L == colmajor ? Eigen::RowMajor : Eigen::ColMajor)>;
	using DType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	using LMapType = Eigen::Map<LType const>;
	using LMapTypeMut = Eigen::Map<LType>;

	using LTMapType = Eigen::Map<LType const>;
	using LTMapTypeMut = Eigen::Map<LType>;

	using DMapType = Eigen::Map<DType const>;
	using DMapTypeMut = Eigen::Map<DType>;

	LType _l;
	DType _d;

public:
	Ldlt(WithDim /*tag*/, i32 dim);

	template <typename D>
	Ldlt(Decompose /*tag*/, Eigen::MatrixBase<D> const& mat);

	auto l() const noexcept -> LMapType;
	auto l_mut() noexcept -> LMapTypeMut;
	auto lt() const noexcept -> LTMapType;
	auto lt_mut() noexcept -> LTMapTypeMut;
	auto d() const noexcept -> DMapType;
	auto d_mut() noexcept -> DMapTypeMut;

	auto reconstructed_matrix() const
			-> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

	template <typename D>
	void decompose(Eigen::MatrixBase<D> const& mat);

	template <typename D>
	void rank_one_update(Eigen::MatrixBase<D> const& w, T alpha);

	// M_new = M + diag(D)
	//
	// for QP
	// D = 1/µ_old - 1/µ_new
	template <typename D>
	void diagonal_update(Eigen::MatrixBase<D> const& diff);

	template <typename Out, typename In>
	void solve( //
			Eigen::DenseCoeffsBase<Out>& x,
			Eigen::MatrixBase<In> const& rhs) const;

	template <typename Out, typename In>
	void solve_in_place(Eigen::DenseCoeffsBase<Out>& rhs) const {
		this->solve(rhs, rhs);
	}

	template <typename D>
	void append_row(Eigen::MatrixBase<D> const& new_row);

	void remove_row(i32 idx);
};

template <typename Scalar>
struct IdentityPreconditionner {
	template <typename Out, typename In>
	void apply_in_place(Eigen::DenseCoeffsBase<Out>& x) {}
	template <typename Out, typename In>
	void apply_inv_in_place(Eigen::DenseCoeffsBase<Out>& x) {}

	template <typename Out, typename In>
	void apply_to_qp_in_place(QpViewMut<Scalar> qp) {}
	template <typename Out, typename In>
	void apply_inv_to_qp_in_place(QpViewMut<Scalar> qp) {}
};

template < //
		typename Scalar,
		typename Preconditionner = IdentityPreconditionner<Scalar>>
auto solve_qp( //
		QpView<Scalar> qp,
		i32 max_iter,
		Scalar eps_abs,
		Scalar eps_rel,
		Preconditioner precond = Preconditionner{}) -> i32 {

	{ precond.apply_in_place(...); }
}

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_LDLT_HPP_VCVSK3EOS */
