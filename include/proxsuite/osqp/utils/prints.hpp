//
// Copyright (c) 2025 INRIA
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
print_preambule()
{
  print_line();
  std::cout << "OSQP - An operator splitting algorithm for QP programs\n"
            << "(c) Paper - Bartolomeo Stellato, Goran Banjac, Paul Goulart, "
               "Alberto Bemporad and Stephen Boyd\n"
            << "(c) Implementation - Lucas Haubert\n"
            << "Inria Paris 2025\n"
            << std::endl;
  print_line();
}

} // end namespace osqp
} // end namespace proxsuite

#endif /* end of include guard PROXSUITE_OSQP_UTILS_PRINTS_HPP */
