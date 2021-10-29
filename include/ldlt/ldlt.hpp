#ifndef INRIA_LDLT_LDLT_HPP_VCVSK3EOS
#define INRIA_LDLT_LDLT_HPP_VCVSK3EOS

#include <Eigen/Core>
#include "ldlt/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include <iostream>

namespace ldlt {

LDLT_DEFINE_TAG(decompose, Decompose);
LDLT_DEFINE_TAG(reserve_uninit, ReserveUninit);

inline void unimplemented() {
	std::terminate();
}
template <typename T>
inline void dump_array(T const* data, usize len) {
	std::cout << '{';
	for (usize i = 0; i < len; ++i) {
		std::cout << data[i];
		std::cout << ',';
	}
	std::cout << "}\n";
}

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

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, dyn, 1> const>;
	using Perm = Eigen::PermutationWrapper<VecMapISize>;

	ColMat _l;
	Vec _d;
	std::vector<isize> perm;
	std::vector<isize> perm_inv;

public:
	Ldlt(ReserveUninit /*tag*/, isize dim)
			: //
				_l(dim, dim),
				_d(dim),
				perm(usize(dim)),
				perm_inv(usize(dim)) {}
	Ldlt(Decompose /*tag*/, ColMat mat)
			: _l(LDLT_FWD(mat)),
				_d(_l.rows()),
				perm(usize(_l.rows())),
				perm_inv(usize(_l.rows())) {
		this->factorize(_l);
	}

	auto p() -> Perm { return Perm(VecMapISize(perm.data(), _l.rows())); }
	auto pt() -> Perm { return Perm(VecMapISize(perm_inv.data(), _l.rows())); }

	auto reconstructed_matrix() const -> ColMat {
		auto A = (_l * _d.asDiagonal() * _l.transpose()).eval();
		isize dim = _d.rows();
		auto tmp = ColMat(dim, dim);
		for (isize i = 0; i < dim; i++) {
			tmp.row(i) = A.row(perm_inv[usize(i)]);
		}
		for (isize i = 0; i < dim; i++) {
			A.col(i) = tmp.col(perm_inv[usize(i)]);
		}
		return A;
	}

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
				perm.data(), perm_inv.data(), {from_eigen, _l.diagonal()});

		ldlt::detail::apply_permutation_sym_work<T>(
				{from_eigen, _l}, perm.data(), {from_eigen, work}, -1);

		ldlt::factorize(
				LdltViewMut<T>{{from_eigen, _l}, {from_eigen, _d}},
				MatrixView<T, colmajor>{from_eigen, _l});
	}

	void factorize(Eigen::Ref<ColMat const> mat) {
		isize n = mat.rows();
		LDLT_WORKSPACE_MEMORY(work, Uninit, Mat(n, n), LDLT_CACHELINE_BYTES, T);
		factorize_work(mat, work.to_eigen());
	}

	void solve_in_place_work(Eigen::Ref<Vec> rhs, Eigen::Ref<Vec> work) const {
		isize n = rhs.rows();
		ldlt::detail::apply_perm_rows<T>::fn(
				work.data(), 0, rhs.data(), 0, n, 1, perm.data(), 0);
		ldlt::solve(
				{from_eigen, work},
				LdltView<T>{
						{from_eigen, _l},
						{from_eigen, _d},
				},
				{from_eigen, work});
		ldlt::detail::apply_perm_rows<T>::fn(
				rhs.data(), 0, work.data(), 0, n, 1, perm_inv.data(), 0);
	}

	void solve_in_place(Eigen::Ref<Vec> rhs) const {
		isize n = _l.rows();
		LDLT_WORKSPACE_MEMORY(work, Uninit, Vec(n), LDLT_CACHELINE_BYTES, T);
		solve_in_place_work(rhs, work.to_eigen());
	}

	void rank_one_update(Eigen::Ref<Vec const> z, T alpha) {
		LDLT_WORKSPACE_MEMORY(
				z_work, Uninit, Vec(z.rows()), LDLT_CACHELINE_BYTES, T);

		isize n = isize(perm.size());
		for (isize i = 0; i < n; ++i) {
			z_work(i) = z.data()[perm[usize(i)]];
		}
		LdltViewMut<T> ld = {
				{from_eigen, _l},
				{from_eigen, _d},
		};

		rank1_update( //
				ld,
				ld.as_const(),
				z_work.as_const(),
				alpha);
	}

	// TODO: avoid reallocating all the time
	void insert_at(isize i, Eigen::Ref<Vec const> a) {
		// insert row/col at end of matrix
		// modify permutation

		// TODO: choose better insertion slot
		isize n = isize(perm.size());
		// dump_array(perm.data(), perm.size());
		// dump_array(perm_inv.data(), perm_inv.size());

		// FIXME: allow insertions anywhere
		if (i != n) {
			unimplemented();
		}

		{
			LDLT_WORKSPACE_MEMORY(
					permuted_a, Uninit, Vec(n + 1), LDLT_CACHELINE_BYTES, T);
			for (isize k = 0; k < n; ++k) {
				permuted_a(k) = a(perm[usize(k)]);
			}
			permuted_a(n) = a(n);

			auto new_l = ColMat(n + 1, n + 1);
			auto new_d = Vec(n + 1);
			row_append(
					LdltViewMut<T>{
							{from_eigen, new_l},
							{from_eigen, new_d},
					},
					LdltView<T>{
							{from_eigen, _l},
							{from_eigen, _d},
					},
					permuted_a.as_const());

			_l = LDLT_FWD(new_l);
			_d = LDLT_FWD(new_d);
			perm.push_back(n);
			perm_inv.push_back(n);
		}
		// dump_array(perm.data(), perm.size());
		// dump_array(perm_inv.data(), perm_inv.size());
	}

	void delete_at(isize i) {
		// delete corresponding row/col after permutation
		// modify permutation
		// dump_array(perm.data(), perm.size());
		// dump_array(perm_inv.data(), perm_inv.size());

		isize n = isize(perm.size());
		isize perm_i = perm_inv[usize(i)];

		// FIXME: handle general case
		if (i != perm_i) {
			std::cout << i << '\n';
			std::cout << perm_i << '\n';
			unimplemented();
		}

		{
			auto new_l = ColMat(n - 1, n - 1);
			auto new_d = Vec(n - 1);
			row_delete(
					LdltViewMut<T>{
							{from_eigen, new_l},
							{from_eigen, new_d},
					},
					LdltView<T>{
							{from_eigen, _l},
							{from_eigen, _d},
					},
					perm_i);

			_l = LDLT_FWD(new_l);
			_d = LDLT_FWD(new_d);

			perm.erase(perm.begin() + i);
			perm_inv.erase(perm_inv.begin() + i);

			for (isize k = i; k < n - 1; ++k) {
				--perm[usize(k)];
				--perm_inv[usize(k)];
			}
		}
		// dump_array(perm.data(), perm.size());
		// dump_array(perm_inv.data(), perm_inv.size());
	}
};

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_LDLT_HPP_VCVSK3EOS */
