//
// Copyright (c) 2022 INRIA
//
/** \file */

#ifndef PROXSUITE_OSQP_UTILS_PRINTS_HPP
#define PROXSUITE_OSQP_UTILS_PRINTS_HPP

#include <iostream>

namespace proxsuite {
namespace osqp {

inline void
print_line()
{
  std::string the_line = "-----------------------------------------------------"
                         "--------------------------------------------\0";
  std::cout << the_line << "\n" << std::endl;
}

inline void
print_header()
{
  std::cout << "iter    objective    pri res    dua res  \n" << std::endl;
}

inline void
print_preambule()
{
  print_line();
  std::cout << "OSQP - Operator Splitting solver for QP problems\n"
            << "A C++ implementation \n"
            << "(c) Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto "
               "Bemporad, Stephen Boyd\n"
            << "Inria Paris 2025\n"
            << std::endl;
  print_line();
}

} // end namespace osqp
} // end namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_UTILS_PRINTS_HPP */
