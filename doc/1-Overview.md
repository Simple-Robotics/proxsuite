<!--
//
// Copyright (c) 2022 INRIA
// Author: Antoine Bambade, Sarah El Kazdadi, Adrien Taylor, Justin Carpentier
//
-->

<img src="https://github.com/Simple-Robotics/proxsuite/raw/main/doc/images/proxsuite-logo.png" width="700" alt="Proxsuite Logo" style="display: block; margin-left: auto; margin-right: auto;"/>

\section OverviewIntro What is ProxSuite?

ProxSuite is a library which provides efficient solvers for solving constrained programs encountered in robotics using dedicated proximal point based algorithms. ProxSuite is open-source, written in C++ with Python bindings, and distributed under the BSD2 licence. Contributions are welcome.


For the moment, the library offers ProxQP solver, which is a C++ implementation of the [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf) for solving convex QPs. It is planned to release soon [an extension](https://hal.archives-ouvertes.fr/hal-03680510/document) for dealing with non linear inequality constraints as well.


In this doc, you will find the usual description of the library functionalities and a quick tutorial with examples on how to use the API.

\section OverviewInstall How to install ProxSuite?

The full installation procedure can be found [here](5-installation.md).

If you just need the Python bindings, you can directly have access to them through Conda. On systems for which binaries are not provided, installation from source should be straightforward. Every release is validated in main Linux, Mac OS X and Windows distributions.

\section OverviewSimple Simplest ProxQP example with compilation command

We start with a simple program to load ProxQP and use ProxQP solver in order to solve a random generated QP problem. It is given in both C++ and Python cases.

<table class="manual">
  <tr>
    <th>examples/cpp/overview-simple.cpp</th>
    <th>examples/python/overview-simple.py</th>
  </tr>
  <tr>
    <td valign="top">
      \include cpp/overview-simple.cpp
    </td>
    <td valign="top">
      \include python/overview-simple.py
    </td>
  </tr>
</table>

\subsection OverviewSimpleCompile Compiling and running your program

You can compile the C++ version by including ProxSuite and Eigen header directories

\code g++ -std=c++17 examples/cpp/overview-simple.cpp -o overview-simple $(pkg-config --cflags proxsuite)  \endcode

If you are looking for the fastest performance, use the following flags to use SIMDE in proxsuite and tell your
compiler to use the corresponding cpu instruction set
\code g++ -O3 -march=native -DNDEBUG -DPROXSUITE_VECTORIZE -std=c++17 examples/cpp/overview-simple.cpp -o overview-simple $(pkg-config --cflags proxsuite)
\endcode

Once your code is compiled, you might then run it using

\code ./overview-simple \endcode

In Python, just run it:

\code python examples/python/overview-simple.py \endcode

\subsection OverviewSimpleExplain Explanation of the program

This program generates a random convex QP problem and loads ProxQP solver with dense backend for solving it.

We first include the proper files. In C++, we have to include the dense backend API imported from "dense.hpp" file, and a header file "util.hpp" for generating a random QP problem (from test directory as it is used for unit testing the solver). In Python, the library is included by just importing proxsuite_pywrap, and we propose a little function `generated_mixed_qp` for generating convex QP with equality and inequality constraints.

The first paragraph generates the QP problem. In the second paragraph, we first define the object `Qp`, using the dimensions of the problem (i.e., `n` is the dimension of primal variable `x`, `n_eq` the number of equality constraints, and `n_in` the number of inequality constraints). We then define some settings for the solver: the accuracy threshold is set to 1.E-9, and the verbose option is set to false (so no intermediary printings will be displayed). The solver is initialized with the previously generated QP problem with the method `init`. Finally, we call the `solve` method. All the results are stored in `Qp.results`.

\section OverviewPython About Python wrappings

ProxSuite is written in C++, with a full template-based C++ API, for efficiency purposes. All the functionalities are available in C++. Extension of the library should be preferably in C++.

However, C++ efficiency comes with a higher work cost, especially for newcomers. For this reason, all the interface is exposed in Python. We tried to build the Python API as much as possible as a mirror of the C++ interface. The greatest difference is that the C++ interface is proposed using Eigen objects for matrices and vectors, that are exposed as NumPy matrices in Python.

\section OverviewCite How to cite ProxSuite?

Happy with ProxSuite? Please cite us with the following format.

\code
@inproceedings{bambade:hal-03683733,
  TITLE = {{PROX-QP: Yet another Quadratic Programming Solver for Robotics and beyond}},
  AUTHOR = {Antoine Bambade, Sarah El-Kazdadi, Adrien Taylor, Justin Carpentier},
  URL = {https://hal.inria.fr/hal-03683733},
  BOOKTITLE = {{RSS 2022 - Robotics: Science and Systems}},
  ADDRESS = {New York, United States},
  YEAR = {2022},
  MONTH = June,
  PDF = {https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf},
  HAL_ID = {hal-03683733},
  HAL_VERSION = {v1},
}
\endcode

The paper is publicly available in HAL ([ref 03683733](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf)).

\section OverviewConclu Where to go from here?

This documentation is mostly composed of several examples, along with a technical documentation. In the next sections you will find illustrated with examples in C++ and python, (i) ProxQP API methods (with dense and sparse backends) ; the lists of solver's settings and results subclasses ; some recommandation about which backend using according to your needs ; some important remarks about timings (and compilation options for speeding up ProxQP considering your OS architecture); and finally (ii) some examples for using ProxQP without passing by the API (but only through a unique solve function).

The last parts of the documentation describes the differents namespaces, classes and files of the C++ project.
