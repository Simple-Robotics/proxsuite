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
  dense::QP<T> qp(dim, n_eq, n_in);
}
