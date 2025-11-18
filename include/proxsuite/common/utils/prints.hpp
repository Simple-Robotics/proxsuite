//
// Copyright (c) 2022-2025 INRIA
//
/** \file */

#ifndef PROXSUITE_COMMON_UTILS_PRINTS_HPP
#define PROXSUITE_COMMON_UTILS_PRINTS_HPP

#include "proxsuite/common/solvers.hpp"
#include <iostream>

namespace proxsuite {
namespace common {

inline void
print_line()
{
  std::string the_line = "-----------------------------------------------------"
                         "--------------------------------------------\0";
  std::cout << the_line << "\n" << std::endl;
}

inline void
print_preambule(const QPSolver solver)
{
  print_line();
  switch (solver) {
    case QPSolver::PROXQP: {
      std::cout
        << "                              ProxQP - Primal-Dual Proximal QP "
           "Solver\n"
        << "     (c) Antoine Bambade, Sarah El Kazdadi, Fabian Schramm, Adrien "
           "Taylor, and "
           "Justin Carpentier\n"
        << "                                         Inria Paris 2022        \n"
        << std::endl;
      break;
    }
    case QPSolver::OSQP: {
      std::cout
        << "OSQP - An operator splitting algorithm for QP programs\n"
        << "(c) Paper - Bartolomeo Stellato, Goran Banjac, Paul Goulart, "
           "Alberto Bemporad and Stephen Boyd\n"
        << "(c) Implementation - Lucas Haubert\n"
        << "Inria Paris 2025\n"
        << std::endl;
      break;
    }
  }

  print_line();
}

} // end namespace common
} // end namespace proxsuite

#endif /* end of include guard PROXSUITE_COMMON_UTILS_PRINTS_HPP */
