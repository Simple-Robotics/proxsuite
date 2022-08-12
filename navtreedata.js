/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "proxsuite", "index.html", [
    [ "Overview", "index.html", [
      [ "What is ProxSuite?", "index.html#OverviewIntro", null ],
      [ "How to install ProxSuite?", "index.html#OverviewInstall", null ],
      [ "Simplest ProxQP example with compilation command", "index.html#OverviewSimple", [
        [ "Compiling and running your program", "index.html#OverviewSimpleCompile", null ],
        [ "Explanation of the program", "index.html#OverviewSimpleExplain", null ]
      ] ],
      [ "About Python wrappings", "index.html#OverviewPython", null ],
      [ "How to cite ProxSuite?", "index.html#OverviewCite", null ],
      [ "Where to go from here?", "index.html#OverviewConclu", null ]
    ] ],
    [ "ProxQP API with examples", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html", [
      [ "ProxQP unified API for dense and sparse backends", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewAPIstructure", [
        [ "The API structure", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewAPI", null ],
        [ "The init method", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#explanationInitMethod", null ],
        [ "The solve method", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#explanationSolveMethod", null ],
        [ "The update method", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#explanationUpdateMethod", null ]
      ] ],
      [ "The settings subclass", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewSettings", [
        [ "The solver's settings", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewAllSettings", null ],
        [ "The different initial guesses", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewInitialGuess", [
          [ "No initial guess", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewNoInitialGuess", null ],
          [ "Equality constrained initial guess", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewEqualityConstrainedInitialGuess", null ],
          [ "Warm start with previous result", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewWarmStartWithPreviousResult", null ],
          [ "Warm start", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewWarmStart", null ],
          [ "Cold start with previous result", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewColdStartWithPreviousResult", null ]
        ] ]
      ] ],
      [ "The results subclass", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewResults", [
        [ "The info subclass", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewInfoClass", null ],
        [ "The solver's status", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewSolverStatus", null ]
      ] ],
      [ "Which backend to use?", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewWhichBackend", null ],
      [ "Some important remarks when computing timings", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewBenchmark", [
        [ "What do the timings take into account?", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewTimings", null ],
        [ "Architecture options when compiling ProxSuite", "md_doc_2__p_r_o_x_q_p__a_p_i_2__prox_q_p_api.html#OverviewArchitectureOptions", null ]
      ] ]
    ] ],
    [ "ProxQP solve function without API", "md_doc_3__prox_q_p_solve.html", [
      [ "A single solve function for dense and sparse backends", "md_doc_3__prox_q_p_solve.html#OverviewAsingleSolveFunction", null ]
    ] ],
    [ "Linear solvers API with examples", "md_doc_4_linearsolvers.html", [
      [ "Dense linear solver", "md_doc_4_linearsolvers.html#OverviewDense", null ],
      [ "Sparse linear solver", "md_doc_4_linearsolvers.html#OverviewSparse", null ]
    ] ],
    [ "Installation", "md_doc_5_installation.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ],
        [ "Enumerator", "namespacemembers_eval.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerator", "functions_eval.html", null ],
        [ "Related Functions", "functions_rela.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Macros", "globals_defs.html", "globals_defs" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"address_8hpp.html",
"dense_2views_8hpp.html#aa1e58cf321f108b1e03b4785168b3ed2",
"hedley_8ext_8hpp.html#a2fedda788307a1a4ef41d11d1d9bc04c",
"macros_8hpp.html#aae522d4ba2f2382dbadd8b7b7cadcc43a7a839798acc3929f5de2caf19905af09",
"namespaceproxsuite_1_1linalg_1_1dense_1_1__detail.html#ab91142b2926a36262b63de3b76ecd30e",
"namespaceproxsuite_1_1linalg_1_1veg_1_1__detail_1_1__collections.html#ad974291cc7f4acccae3ccbbdcbe15af1",
"namespaceproxsuite_1_1proxqp_1_1concepts.html",
"placement_8hpp.html",
"preprocessor_8hpp.html#af9c976b7311da40c522f8d89573f1db1",
"sparse_2utils_8hpp.html#ac0e5d8bb606e37f5b5708090a2e00b3c",
"structproxsuite_1_1linalg_1_1dense_1_1__detail_1_1_indices_r.html#a37c55623bd6e514db75e4eeec13c33b2",
"structproxsuite_1_1linalg_1_1sparse_1_1_mat_mut.html",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1__collections_1_1_vec_impl.html#a510b969deb0796a66d2aacd7b9bb1a26",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1__meta_1_1array__get.html",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1_move_fn.html",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1meta___1_1_indexed_to_tuple.html",
"structproxsuite_1_1linalg_1_1veg_1_1array_1_1_array.html",
"structproxsuite_1_1linalg_1_1veg_1_1mem_1_1nb_1_1align__next.html",
"structproxsuite_1_1linalg_1_1veg_1_1tuple_1_1nb_1_1tuplify.html#a935cadc529630478102c77c28582b631",
"structproxsuite_1_1proxqp_1_1_settings.html#af3ab3bd10e449f60b8e0c60c636363b1",
"structproxsuite_1_1proxqp_1_1dense_1_1_workspace.html",
"structproxsuite_1_1proxqp_1_1detail_1_1_element_access_3_01_layout_1_1rowmajor_01_4.html#a680dff2a8f72be532dc7cc369caa16e1",
"structproxsuite_1_1proxqp_1_1sparse_1_1_workspace.html#ab43c53d9268564e206073c557afaf9c3",
"veg_2type__traits_2core_8hpp.html#a3dad0e0cd361b02d919919ab4e54c462"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';