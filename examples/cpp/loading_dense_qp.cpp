#include <iostream>
#include "proxsuite/proxqp/dense/dense.hpp"

using T = double;
using namespace proxsuite;
using namespace proxsuite::common;

int
main()
{
  isize dim = 10;
  isize n_eq(dim / 4);
  isize n_in(dim / 4);
  proxqp::dense::QP<T> qp(dim, n_eq, n_in);
}
