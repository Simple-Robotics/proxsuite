/** \file */

#ifndef PROXSUITE_QP_SPARSE_VIEWS_HPP
#define PROXSUITE_QP_SPARSE_VIEWS_HPP

#include <linearsolver/dense/core.hpp>
#include <linearsolver/sparse/core.hpp>
#include <linearsolver/sparse/factorize.hpp>
#include <linearsolver/sparse/update.hpp>
#include <linearsolver/sparse/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/settings.hpp>
#include <veg/vec.hpp>
#include "qp/sparse/data.hpp"
#include "qp/results.hpp"

#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace qp {
namespace sparse {

template <typename T, typename I>
struct QpView {
	linearsolver::sparse::MatRef<T, I> H;
	linearsolver::sparse::DenseVecRef<T> g;
	linearsolver::sparse::MatRef<T, I> AT;
	linearsolver::sparse::DenseVecRef<T> b;
	linearsolver::sparse::MatRef<T, I> CT;
	linearsolver::sparse::DenseVecRef<T> l;
	linearsolver::sparse::DenseVecRef<T> u;
};

template <typename T, typename I>
struct QpViewMut {
	linearsolver::sparse::MatMut<T, I> H;
	linearsolver::sparse::DenseVecMut<T> g;
	linearsolver::sparse::MatMut<T, I> AT;
	linearsolver::sparse::DenseVecMut<T> b;
	linearsolver::sparse::MatMut<T, I> CT;
	linearsolver::sparse::DenseVecMut<T> l;
	linearsolver::sparse::DenseVecMut<T> u;

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
}//namespace qp
}//namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_WORKSPACE_HPP */