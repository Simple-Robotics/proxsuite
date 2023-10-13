#include <iostream>
#include "proxsuite/proxqp/dense/dense.hpp"

using namespace proxsuite::proxqp;
using T = double;
int
main()
{
  dense::isize dim = 10;
  dense::isize n_eq(dim / 4);
  dense::isize n_in(dim / 4);
  // we load here a QP model with
  // n_eq equality constraints
  // n_in generic type of inequality constraints
  // and dim box inequality constraints
  // we specify PrimalDualLDLT backend
  dense::QP<T> qp(
    dim, n_eq, n_in, true, proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
  // true specifies we take into accounts box constraints
  // n_in are any other type of inequality constraints

  // Other examples

  // we load here a QP model with
  // n_eq equality constraints
  // O generic type of inequality constraints
  // and dim box inequality constraints
  // we specify PrimalLDLT backend
  dense::QP<T> qp2(
    dim, n_eq, 0, true, proxsuite::proxqp::DenseBackend::PrimalLDLT);
  // true specifies we take into accounts box constraints
  // we don't need to precise n_in = dim, it is taken
  // into account internally

  // we let here the solver decide with Automatic
  // backend choice.
  dense::QP<T> qp3(
    dim, n_eq, 0, true, proxsuite::proxqp::DenseBackend::Automatic);
  // Note that it is the default choice
}
