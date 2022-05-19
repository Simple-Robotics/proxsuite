/** \file */
#ifndef PROXSUITE_QP_SPARSE_DATA_HPP
#define PROXSUITE_QP_SPARSE_DATA_HPP

#include "linearsolver/sparse/core.hpp"
#include <veg/vec.hpp>

namespace proxsuite {
namespace qp {
namespace sparse {

using veg::isize;
using veg::usize;
using veg::i64;

template <typename T, typename I>
struct Data {
	isize dim;
	isize n_eq;
	isize n_in;

  isize H_nnz;
  isize A_nnz;
  isize C_nnz;

	veg::Vec<I> kkt_col_ptrs;
	veg::Vec<I> kkt_row_indices;
	veg::Vec<T> kkt_values;
  Eigen::Matrix<T, Eigen::Dynamic, 1> g;
  Eigen::Matrix<T, Eigen::Dynamic, 1> b;
  Eigen::Matrix<T, Eigen::Dynamic, 1> l;
  Eigen::Matrix<T, Eigen::Dynamic, 1> u;

	auto kkt() const -> linearsolver::sparse::MatMut<T, I> {
		auto n_tot = kkt_col_ptrs.len() - 1;
		auto nnz =
				isize(linearsolver::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
		return {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs.ptr(),
				nullptr,
				kkt_row_indices.ptr(),
				kkt_values.ptr(),
		};
	}
	auto kkt_mut() -> linearsolver::sparse::MatMut<T, I> {
		auto n_tot = kkt_col_ptrs.len() - 1;
		auto nnz =
				isize(linearsolver::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
		return {
				linearsolver::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				kkt_col_ptrs.ptr_mut(),
				nullptr,
				kkt_row_indices.ptr_mut(),
				kkt_values.ptr_mut(),
		};
	}
};

} // namespace sparse
} // namespace qp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_DATA_HPP */
