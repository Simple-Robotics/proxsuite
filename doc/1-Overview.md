# Overview {#index}
<!--
//
// Copyright (c) 2022 INRIA
// Author: Antoine Bambade, Sarah El Kazdadi, Adrien Taylor, Justin Carpentier
//
-->

\section OverviewIntro What is ProxSuite ? (TO FINISH)

ProxSuite is a library which provides efficient solvers for solving constrained programs (convex QPs, QCQPs etc.) encountered in robotics using dedicated proximal point based algorithms.

For convex QPs, ProxQP is a C++ implementation of the [ProxQP algorithm](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf).

ProxSuite is open-source, written in C++ with Python bindings, and distributed under the BSD2 licence.
Contributions are welcome.

In this doc, you will find the usual description of the library functionalities, a quick tutorial with examples on how to use the API.

\section OverviewPython About Python wrappings

ProxSuite is written in C++, with a full template-based C++ API, for efficiency purposes. All the functionalities are available in C++. Extension of the library should be preferably in C++.

However, C++ efficiency comes with a higher work cost, especially for newcomers. For this reason, all the interface is exposed in Python. We tried to build the Python API as much as possible as a mirror of the C++ interface. The greatest difference is that the C++ interface is proposed using Eigen objects for matrices and vectors, that are exposed as NumPy matrices in Python.

\section OverviewCite How to cite ProxSuite

Happy with ProxSuite? Please cite us with the following format.

### Easy solution: cite our open access paper
The following is the preferred way to cite ProxQP.
The paper is publicly available in HAL ([ref 03683733](https://hal.inria.fr/hal-03683733/file/Yet_another_QP_solver_for_robotics_and_beyond.pdf)).