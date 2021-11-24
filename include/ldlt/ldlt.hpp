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

template <typename T>
struct Ldlt {
private:
	static constexpr auto DYN = Eigen::Dynamic;
	using ColMat = Eigen::Matrix<T, DYN, DYN, Eigen::ColMajor>;
	using RowMat = Eigen::Matrix<T, DYN, DYN, Eigen::RowMajor>;
	using Vec = Eigen::Matrix<T, DYN, 1>;

	using LView = Eigen::TriangularView<
			Eigen::Map< //
					ColMat const,
					Eigen::Aligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitLower>;
	using LViewMut = Eigen::TriangularView<
			Eigen::Map< //
					ColMat,
					Eigen::Aligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitLower>;

	using LTView = Eigen::TriangularView<
			Eigen::Map< //
					RowMat const,
					Eigen::Aligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitUpper>;
	using LTViewMut = Eigen::TriangularView<
			Eigen::Map< //
					RowMat,
					Eigen::Aligned,
					Eigen::OuterStride<DYN>>,
			Eigen::UnitUpper>;

	using ColMatMap = Eigen::Map<ColMat const, Eigen::Aligned, Eigen::OuterStride<DYN>>;
	using ColMatMapMut = Eigen::Map<ColMat, Eigen::Aligned, Eigen::OuterStride<DYN>>;

	using RowMatMap = Eigen::Map<RowMat const, Eigen::Aligned, Eigen::OuterStride<DYN>>;
	using RowMatMapMut = Eigen::Map<RowMat, Eigen::Aligned, Eigen::OuterStride<DYN>>;

	using VecMap = Eigen::Map<Vec const, Eigen::Aligned>;
	using VecMapMut = Eigen::Map<Vec, Eigen::Aligned>;

	using VecMapISize = Eigen::Map<Eigen::Matrix<isize, DYN, 1> const>;
	using Perm = Eigen::PermutationWrapper<VecMapISize>;

	ColMat _l, _new_l;
	Vec _d, _new_d, _permuted_a;
	std::vector<isize> perm;
	std::vector<isize> perm_inv;
	isize dim, max_dim;

public:
  
  virtual ~Ldlt()
  {
//    std::cout << "_d" << _d.transpose() << std::endl;
//    _l.~ColMat();
  }
	Ldlt(ReserveUninit /*tag*/, isize dim, isize _max_dim = 0)
			: //
				dim(dim) 
	{
		if(_max_dim == 0)
			max_dim = dim;
		else
			max_dim = _max_dim;

		_l.resize(max_dim,max_dim);
    _new_l.resize(max_dim,max_dim);
		_d.resize(max_dim);
    _new_d.resize(max_dim);
    _permuted_a.resize(max_dim);

		perm.reserve(usize(max_dim)); perm.resize(usize(dim));
		perm_inv.reserve(usize(max_dim)); perm_inv.resize(usize(dim));
	}

	Ldlt(Decompose /*tag*/, ColMat mat, isize _max_dim = 0)
			: //
				dim(mat.rows()) 
	{
		if(_max_dim == 0)
			max_dim = dim;
		else
			max_dim = _max_dim;

		_l.resize(max_dim,max_dim);
		_l.topLeftCorner(dim,dim) = mat;
    _new_l.resize(max_dim,max_dim);
		_d.resize(max_dim);
    _new_d.resize(max_dim);
    _permuted_a.resize(max_dim);

		perm.reserve(usize(max_dim)); perm.resize(usize(dim));
		perm_inv.reserve(usize(max_dim)); perm_inv.resize(usize(dim));
		
		this->factorize(_l);
	}

	auto p() -> Perm { return Perm(VecMapISize(perm.data(), l().rows())); }
	auto pt() -> Perm { return Perm(VecMapISize(perm_inv.data(), l().rows())); }

	auto reconstructed_matrix() const -> ColMat {
		isize n = _d.rows();
		auto tmp = ColMat(n, n);
		tmp = ltri();
		tmp = tmp * _d.asDiagonal();
		auto A = ColMat(tmp * ltrit());

		for (isize i = 0; i < n; i++) {
			tmp.row(i) = A.row(perm_inv[usize(i)]);
		}
		for (isize i = 0; i < n; i++) {
			A.col(i) = tmp.col(perm_inv[usize(i)]);
		}
		return A;
	}

	auto l() const noexcept -> ColMatMap {
		return Eigen::Map< //
							 ColMat const,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride());
	}

	auto l_mut() noexcept -> ColMatMapMut {
		return Eigen::Map< //
							 ColMat,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride());
	}
  
  auto new_l_mut(isize n) noexcept -> ColMatMapMut {
    return Eigen::Map< //
               ColMat,
               Eigen::Aligned,
               Eigen::OuterStride<DYN>>(
               _new_l.data(), n, n, _new_l.outerStride());
  }

	auto ltri() const noexcept -> LView {
		return Eigen::Map< //
							 ColMat const,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride())
		    .template triangularView<Eigen::UnitLower>();
	}
	auto ltri_mut() noexcept -> LViewMut {
		return Eigen::Map< //
							 ColMat,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride())
		    .template triangularView<Eigen::UnitLower>();
	}
	auto lt() const noexcept -> RowMatMap {
		return Eigen::Map< //
							 RowMat const,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride());
	}
	auto ltrit() const noexcept -> LTView {
		return Eigen::Map< //
							 RowMat const,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride())
		    .template triangularView<Eigen::UnitUpper>();
	}
	auto lt_mut() noexcept -> LTViewMut {
		return Eigen::Map< //
							 RowMat,
							 Eigen::Aligned,
							 Eigen::OuterStride<DYN>>(
							 _l.data(), dim, dim, _l.outerStride())
		    .template triangularView<Eigen::UnitUpper>();
	}
	auto d() const noexcept -> VecMap { return VecMap(_d.data(), dim); }
	auto d_mut() noexcept -> VecMapMut { return VecMapMut(_d.data(), dim); }
  auto new_d(isize n) const noexcept -> VecMap { return VecMap(_new_d.data(), n); }
  auto new_d_mut(isize n) noexcept -> VecMapMut { return VecMapMut(_new_d.data(), n); }
  auto permuted_a_mut(isize n) noexcept -> VecMapMut { return VecMapMut(_permuted_a.data(), n); }
  auto permuted_a(isize n) const noexcept -> VecMap { return VecMap(_permuted_a.data(), n); }


	void factorize_work(Eigen::Ref<ColMat const> mat, Eigen::Ref<ColMat> work) {
		isize n = mat.rows();
		if (_l.rows() != mat.rows()) {
			dim = n;
			max_dim = std::max(n,max_dim);

			_d.conservativeResize(max_dim);
			_l.conservativeResize(max_dim, max_dim);
			perm.resize(usize(n));
			perm_inv.resize(usize(n));
		}
		if (_l.data() != mat.data()) {
			l_mut() = mat;
		}

		ldlt::detail::compute_permutation<T>(
				perm.data(), perm_inv.data(), {from_eigen, l_mut().diagonal()});

		ldlt::detail::apply_permutation_sym_work<T>(
				{from_eigen, l_mut()}, perm.data(), {from_eigen, work}, -1);

		ldlt::factorize(
				LdltViewMut<T>{{from_eigen, l_mut()}, {from_eigen, d_mut()}},
				MatrixView<T, colmajor>{from_eigen, l_mut()});
	}

	void factorize(Eigen::Ref<ColMat const> mat) {
		isize n = mat.rows();
		// LDLT_WORKSPACE_MEMORY(work, Uninit, Mat(n, n), LDLT_CACHELINE_BYTES, T);
		// factorize_work(mat, work.to_eigen());
		auto work = new_l_mut(n);
		factorize_work(mat, work);
	}

	void solve_in_place_work(Eigen::Ref<Vec> rhs, Eigen::Ref<Vec> work) const {
		isize n = rhs.rows();
		ldlt::detail::apply_perm_rows<T>::fn(
				work.data(), 0, rhs.data(), 0, n, 1, perm.data(), 0);
		ldlt::solve(
				{from_eigen, work},
				LdltView<T>{
						{from_eigen, l()},
						{from_eigen, d()},
				},
				{from_eigen, work});
		ldlt::detail::apply_perm_rows<T>::fn(
				rhs.data(), 0, work.data(), 0, n, 1, perm_inv.data(), 0);
	}

	void solve_in_place(Eigen::Ref<Vec> rhs) const {
		isize n = l().rows();
		// LDLT_WORKSPACE_MEMORY(work, Uninit, Vec(n), LDLT_CACHELINE_BYTES, T);
		auto work = const_cast<Ldlt&>(*this).new_d_mut(n);
		// solve_in_place_work(rhs, work.to_eigen());
		solve_in_place_work(rhs, work);
	}

	void rank_one_update(Eigen::Ref<Vec const> z, T alpha) {
		// LDLT_WORKSPACE_MEMORY(
		// 		z_work, Uninit, Vec(z.rows()), LDLT_CACHELINE_BYTES, T);

		auto z_work = new_d_mut(z.rows());
		isize n = isize(perm.size());
		for (isize i = 0; i < n; ++i) {
			z_work(i) = z.data()[perm[usize(i)]];
		}
		LdltViewMut<T> ld = {
				{from_eigen, l_mut()},
				{from_eigen, d_mut()},
		};

		rank1_update( //
				ld,
				ld.as_const(),
				// z_work.as_const(),
				{from_eigen,z_work},
				alpha);
	}

	// TODO: avoid reallocating all the time
	void insert_at(isize i, Eigen::Ref<Vec const> a) {
		// insert row/col at end of matrix
		// modify permutation

		// TODO: choose better insertion slot
		isize n = isize(perm.size());

		isize i_actual = n;

		{
//			LDLT_WORKSPACE_MEMORY(
//					permuted_a, Uninit, Vec(n + 1), LDLT_CACHELINE_BYTES, T);

//			LDLT_WORKSPACE_MEMORY(
//					new_l, Uninit, Mat(n + 1, n +1), LDLT_CACHELINE_BYTES, T);
//			LDLT_WORKSPACE_MEMORY(
//					new_d, Uninit, Vec(n + 1), LDLT_CACHELINE_BYTES, T);

			for (isize k = 0; k < n; ++k) {
				auto& p_k = perm[usize(k)];
				auto& pinv_k = perm_inv[usize(k)];

				if (p_k >= i) {
					++p_k;
				}
				if (pinv_k >= i_actual) {
					++pinv_k;
				}
			}

			perm.insert(perm.begin() + i_actual, i);
			perm_inv.insert(perm_inv.begin() + i, i_actual);

      auto permuted_a = permuted_a_mut(n+1);
			for (isize k = 0; k < n + 1; ++k) {
				permuted_a(k) = a(perm[usize(k)]);
			}

			// auto new_l = ColMat(n + 1, n + 1);
			// auto new_d = Vec(n + 1);
      
			row_append(
					LdltViewMut<T>{
        // {from_eigen, new_l},
        // {from_eigen, new_d},
//							{from_eigen, new_l.to_eigen()},
//							{from_eigen, new_d.to_eigen()},
        {from_eigen, new_l_mut(n+1)},
        {from_eigen, new_d_mut(n+1)},
					},
					LdltView<T>{
							{from_eigen, l_mut()},
							{from_eigen, d_mut()},
					},
                 {from_eigen, _permuted_a.head(n+1)}
                 );

			dim += 1;
      // l_mut() = LDLT_FWD(new_l);
      // d_mut() = LDLT_FWD(new_d);
//			l_mut() = new_l.to_eigen();
//			d_mut() = new_d.to_eigen();
      l_mut() = new_l_mut(n+1);
      d_mut() = new_d_mut(n+1);
		}
	}

	void delete_at(isize i) {
		// delete corresponding row/col after permutation
		// modify permutation

		isize n = isize(perm.size());
		isize i_actual = perm_inv[usize(i)];

//		LDLT_WORKSPACE_MEMORY(
//			new_l, Uninit, Mat(n - 1, n - 1), LDLT_CACHELINE_BYTES, T);
//		LDLT_WORKSPACE_MEMORY(
//			new_d, Uninit, Vec(n - 1), LDLT_CACHELINE_BYTES, T);

		{
			// auto new_l = ColMat(n - 1, n - 1);
			// auto new_d = Vec(n - 1);

			row_delete(
					LdltViewMut<T>{
        // {from_eigen, new_l},
        // {from_eigen, new_d},
//							{from_eigen, new_l.to_eigen()},
//							{from_eigen, new_d.to_eigen()},
        {from_eigen, new_l_mut(n-1)},
        {from_eigen, new_d_mut(n-1)},
					},
					LdltView<T>{
							{from_eigen, l_mut()},
							{from_eigen, d_mut()},
					},
					i_actual);

//      std::cout << "new_l: " << new_l.to_eigen()(0,0) << std::endl;
//      std::cout << "new_d: " << new_d.to_eigen()[0] << std::endl;
			dim -= 1;
      // l_mut() = LDLT_FWD(new_l);
      // d_mut() = LDLT_FWD(new_d);
      //      l_mut() = new_l.to_eigen();
      //      d_mut() = new_d.to_eigen();
      l_mut() = new_l_mut(n-1);
      d_mut() = new_d_mut(n-1);

			perm.erase(perm.begin() + i_actual);
			perm_inv.erase(perm_inv.begin() + i);

			for (isize k = 0; k < n - 1; ++k) {
				auto& p_k = perm[usize(k)];
				auto& pinv_k = perm_inv[usize(k)];

				if (p_k > i) {
					--p_k;
				}
				if (pinv_k > i_actual) {
					--pinv_k;
				}
			}
		}
	}
};

} // namespace ldlt

#endif /* end of include guard INRIA_LDLT_LDLT_HPP_VCVSK3EOS */
