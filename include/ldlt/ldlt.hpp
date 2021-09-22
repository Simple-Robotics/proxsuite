#ifndef INRIA_LDLT_LDLT_HPP_VCVSK3EOS
#define INRIA_LDLT_LDLT_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/solve.hpp"

namespace ldlt {

LDLT_DEFINE_TAG(decompose, Decompose);
LDLT_DEFINE_TAG(reserve_uninit, ReserveUninit);

namespace detail {
template <typename T>
auto make_unique_array(isize n) -> std::unique_ptr<T[]> {
	return std::unique_ptr<T[]>(new T[usize(n)]{});
}
template <typename T>
auto make_unique_array_uninit(isize n) -> std::unique_ptr<T[]> {
	return std::unique_ptr<T[]>(new T[usize(n)]);
}
} // namespace detail

template <typename T>
struct Ldlt {
private:
	static constexpr auto dyn = Eigen::Dynamic;
	using ColMat = Eigen::Matrix<T, dyn, dyn, Eigen::ColMajor>;
	using RowMat = Eigen::Matrix<T, dyn, dyn, Eigen::RowMajor>;
	using Vec = Eigen::Matrix<T, dyn, 1>;

	using LView = Eigen::TriangularView<
			Eigen::Map< //
					ColMat const,
					Eigen::Unaligned,
					Eigen::OuterStride<dyn>>,
			Eigen::StrictlyLower>;
	using LViewMut = Eigen::TriangularView<
			Eigen::Map< //
					ColMat,
					Eigen::Unaligned,
					Eigen::OuterStride<dyn>>,
			Eigen::StrictlyLower>;

	using LTView = Eigen::TriangularView<
			Eigen::Map< //
					RowMat const,
					Eigen::Unaligned,
					Eigen::OuterStride<dyn>>,
			Eigen::StrictlyUpper>;
	using LTViewMut = Eigen::TriangularView<
			Eigen::Map< //
					RowMat,
					Eigen::Unaligned,
					Eigen::OuterStride<dyn>>,
			Eigen::StrictlyUpper>;

	using VecMap = Eigen::Map<Vec const>;
	using VecMapMut = Eigen::Map<Vec>;

	using VecMapI32 = Eigen::Map<Eigen::Matrix<i32, dyn, 1> const>;
	using Perm = Eigen::PermutationWrapper<VecMapI32>;

	ColMat _l;
	Vec _d;
	std::unique_ptr<i32[]> perm;
	std::unique_ptr<i32[]> perm_inv;

public:
	Ldlt(ReserveUninit /*tag*/, isize dim)
			: _l(dim, dim),
				_d(dim),
				perm(detail::make_unique_array_uninit<i32>(dim)),
				perm_inv(detail::make_unique_array_uninit<i32>(dim)) {}
	Ldlt(Decompose /*tag*/, ColMat mat)
			: _l(LDLT_FWD(mat)),
				_d(_l.rows()),
				perm(detail::make_unique_array_uninit<i32>(_l.rows())),
				perm_inv(detail::make_unique_array_uninit<i32>(_l.rows())) {
		this->factorize(_l);
	}

	auto p() -> Perm { return Perm(VecMapI32(perm.get(), _l.rows())); }
	auto pt() -> Perm { return Perm(VecMapI32(perm_inv.get(), _l.rows())); }

	auto l() const noexcept -> LView {
		return Eigen::Map< //
							 ColMat const,
							 Eigen::Unaligned,
							 Eigen::OuterStride<dyn>>(
							 _l.data(), _l.rows(), _l.cols(), _l.outerStride())
		    .template triangularView<Eigen::StrictlyLower>();
	}
	auto l_mut() noexcept -> LViewMut {
		return Eigen::Map< //
							 ColMat,
							 Eigen::Unaligned,
							 Eigen::OuterStride<dyn>>(
							 _l.data(), _l.rows(), _l.cols(), _l.outerStride())
		    .template triangularView<Eigen::StrictlyLower>();
	}
	auto lt() const noexcept -> LTView {
		return Eigen::Map< //
							 RowMat const,
							 Eigen::Unaligned,
							 Eigen::OuterStride<dyn>>(
							 _l.data(), _l.rows(), _l.cols(), _l.outerStride())
		    .template triangularView<Eigen::StrictlyUpper>();
	}
	auto lt_mut() noexcept -> LTViewMut {
		return Eigen::Map< //
							 RowMat,
							 Eigen::Unaligned,
							 Eigen::OuterStride<dyn>>(
							 _l.data(), _l.rows(), _l.cols(), _l.outerStride())
		    .template triangularView<Eigen::StrictlyUpper>();
	}
	auto d() const noexcept -> VecMap { return VecMap(_d.data(), _d.rows()); }
	auto d_mut() noexcept -> VecMapMut { return VecMapMut(_d.data(), _d.rows()); }

	void factorize_work(Eigen::Ref<ColMat const> mat, Eigen::Ref<ColMat> work) {
		isize n = mat.rows();
		if (_l.rows() != mat.rows()) {
			_d.resize(n);
			_l.resize(n, n);
		}
		if (_l.data() != mat.data()) {
			_l = mat;
		}

		ldlt::detail::compute_permutation<T>(
				perm.get(), perm_inv.get(), {from_eigen, _l.diagonal()});

		ldlt::detail::apply_permutation_sym_work<T>(
				{from_eigen, _l}, perm.get(), {from_eigen, work}, -1);

		ldlt::factorize(
				LdltViewMut<T>{{from_eigen, _l}, {from_eigen, _d}},
				MatrixView<T, colmajor>{from_eigen, _l});
	}

	void factorize(Eigen::Ref<ColMat const> mat) {
		isize n = mat.rows();
		LDLT_WORKSPACE_MEMORY(work, Mat(n, n), T);
		factorize_work(mat, work.to_eigen());
	}

	void solve_in_place_work(Eigen::Ref<Vec> rhs, Eigen::Ref<Vec> work) const {
		isize n = rhs.rows();
		ldlt::detail::apply_perm_rows<T>::fn(
				work.data(), 0, rhs.data(), 0, n, 1, perm.get(), 0);
		ldlt::solve(
				{from_eigen, work},
				LdltView<T>{
						{from_eigen, _l},
						{from_eigen, _d},
				},
				{from_eigen, work});
		ldlt::detail::apply_perm_rows<T>::fn(
				rhs.data(), 0, work.data(), 0, n, 1, perm_inv.get(), 0);
	}

	void solve_in_place(Eigen::Ref<Vec> rhs) const {
		isize n = _l.rows();
		LDLT_WORKSPACE_MEMORY(work, Vec(n), T);
		solve_in_place_work(rhs, work.to_eigen());
	}
};

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_LDLT_HPP_VCVSK3EOS */
