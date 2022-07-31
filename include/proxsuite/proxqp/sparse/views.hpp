//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_QP_SPARSE_VIEWS_HPP
#define PROXSUITE_QP_SPARSE_VIEWS_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/sparse/core.hpp>
#include <proxsuite/linalg/sparse/factorize.hpp>
#include <proxsuite/linalg/sparse/update.hpp>
#include <proxsuite/linalg/sparse/rowmod.hpp>
#include <proxsuite/proxqp/dense/views.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/veg/vec.hpp>
#include "proxsuite/proxqp/sparse/model.hpp"
#include "proxsuite/proxqp/results.hpp"

#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace proxqp {
namespace sparse {

template <typename T, typename I>
struct QpView {
	linalg::sparse::MatRef<T, I> H;
	linalg::sparse::DenseVecRef<T> g;
	linalg::sparse::MatRef<T, I> AT;
	linalg::sparse::DenseVecRef<T> b;
	linalg::sparse::MatRef<T, I> CT;
	linalg::sparse::DenseVecRef<T> l;
	linalg::sparse::DenseVecRef<T> u;
};

template <typename T, typename I>
struct QpViewMut {
	linalg::sparse::MatMut<T, I> H;
	linalg::sparse::DenseVecMut<T> g;
	linalg::sparse::MatMut<T, I> AT;
	linalg::sparse::DenseVecMut<T> b;
	linalg::sparse::MatMut<T, I> CT;
	linalg::sparse::DenseVecMut<T> l;
	linalg::sparse::DenseVecMut<T> u;

	auto as_const() noexcept -> QpView<T, I> {
		return {
				H.as_const(),
				g.as_const(),
				AT.as_const(),
				b.as_const(),
				CT.as_const(),
				l.as_const(),
				u.as_const(),
		};
	}
};

}//namespace sparse
}//namespace proxqp
}//namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WORKSPACE_HPP */
