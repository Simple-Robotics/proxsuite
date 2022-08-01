//
// Copyright (c) 2022 INRIA
//
/** \file */
#ifndef PROXSUITE_QP_SPARSE_MODEL_HPP
#define PROXSUITE_QP_SPARSE_MODEL_HPP

#include "proxsuite/linalg/sparse/core.hpp"
#include "proxsuite/proxqp/sparse/fwd.hpp"

namespace proxsuite {
namespace proxqp {
namespace sparse {
///
/// @brief This class stores the model of the QP problem.
///
/*!
 * Model class of the sparse solver storing the QP problem structure.
*/
template <typename T, typename I>
struct Model {
	isize dim;
	isize n_eq;
	isize n_in;

	isize H_nnz;
	isize A_nnz;
	isize C_nnz;

	proxsuite::linalg::veg::Vec<I> kkt_col_ptrs;
	proxsuite::linalg::veg::Vec<I> kkt_row_indices; 
	proxsuite::linalg::veg::Vec<T> kkt_values;

	proxsuite::linalg::veg::Vec<I> kkt_col_ptrs_unscaled;
	proxsuite::linalg::veg::Vec<I> kkt_row_indices_unscaled; 
	proxsuite::linalg::veg::Vec<T> kkt_values_unscaled;

	Eigen::Matrix<T, Eigen::Dynamic, 1> g;
	Eigen::Matrix<T, Eigen::Dynamic, 1> b;
	Eigen::Matrix<T, Eigen::Dynamic, 1> l;
	Eigen::Matrix<T, Eigen::Dynamic, 1> u;

	/*!
	 * Returns the current (scaled) KKT matrix of the problem.
	 */
	auto kkt() const -> proxsuite::linalg::sparse::MatRef<T, I> {
		auto n_tot = kkt_col_ptrs.len() - 1;
		auto nnz =
				isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
		return {
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs.ptr(),
				nullptr,
				kkt_row_indices.ptr(),
				kkt_values.ptr(),
		};
	}
	/*!
	 * Returns the current (scaled) KKT matrix of the problem (mutable form).
	 */
	auto kkt_mut() -> proxsuite::linalg::sparse::MatMut<T, I> {
		auto n_tot = kkt_col_ptrs.len() - 1;
		auto nnz =
				isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
		return {
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs.ptr_mut(),
				nullptr,
				kkt_row_indices.ptr_mut(),
				kkt_values.ptr_mut(),
		};
	}
	/*!
	 * Returns the original (unscaled) KKT matrix of the problem.
	 */
	auto kkt_unscaled() const -> proxsuite::linalg::sparse::MatRef<T, I> {
		auto n_tot = kkt_col_ptrs_unscaled.len() - 1;
		auto nnz =
				isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs_unscaled[n_tot]));
		return {
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs_unscaled.ptr(),
				nullptr,
				kkt_row_indices_unscaled.ptr(),
				kkt_values_unscaled.ptr(),
		};
	}
	/*!
	 * Returns the original (unscaled) KKT matrix of the problem (mutable form).
	 */
	auto kkt_mut_unscaled() -> proxsuite::linalg::sparse::MatMut<T, I> {
		auto n_tot = kkt_col_ptrs_unscaled.len() - 1;
		auto nnz =
				isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs_unscaled[n_tot]));
		return {
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs_unscaled.ptr_mut(),
				nullptr,
				kkt_row_indices_unscaled.ptr_mut(),
				kkt_values_unscaled.ptr_mut(),
		};
	}

};

} // namespace sparse
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_MODEL_HPP */
