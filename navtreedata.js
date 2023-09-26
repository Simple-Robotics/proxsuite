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
    [ "What is ProxSuite?", "index.html#OverviewIntro", null ],
    [ "How to install ProxSuite?", "index.html#OverviewInstall", null ],
    [ "Simplest ProxQP example with compilation command", "index.html#OverviewSimple", [
      [ "Compiling and running your program", "index.html#OverviewSimpleCompile", null ],
      [ "Explanation of the program", "index.html#OverviewSimpleExplain", null ]
    ] ],
    [ "About Python wrappings", "index.html#OverviewPython", null ],
    [ "How to cite ProxSuite?", "index.html#OverviewCite", null ],
    [ "Where to go from here?", "index.html#OverviewConclu", null ],
    [ "ProxQP API with examples", "md_doc_22-PROXQP__API_22-ProxQP__api.html", [
      [ "ProxQP unified API for dense and sparse backends", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewAPIstructure", [
        [ "The API structure", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewAPI", null ],
        [ "The init method", "md_doc_22-PROXQP__API_22-ProxQP__api.html#explanationInitMethod", null ],
        [ "The solve method", "md_doc_22-PROXQP__API_22-ProxQP__api.html#explanationSolveMethod", null ],
        [ "The update method", "md_doc_22-PROXQP__API_22-ProxQP__api.html#explanationUpdateMethod", null ]
      ] ],
      [ "The settings subclass", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewSettings", [
        [ "The solver's settings", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewAllSettings", null ],
        [ "The different initial guesses", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewInitialGuess", [
          [ "No initial guess", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewNoInitialGuess", null ]
        ] ],
        [ "The different options for estimating H minimal Eigenvalue", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewEstimatingHminimalEigenValue", [
          [ "Equality constrained initial guess", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewEqualityConstrainedInitialGuess", null ],
          [ "Warm start with the previous result", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewWarmStartWithPreviousResult", null ],
          [ "Warm start", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewWarmStart", null ],
          [ "Cold start with previous result", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewColdStartWithPreviousResult", null ]
        ] ]
      ] ],
      [ "The results subclass", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewResults", [
        [ "The info subclass", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewInfoClass", null ],
        [ "The solver's status", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewSolverStatus", null ]
      ] ],
      [ "Which backend to use?", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewWhichBackend", null ],
      [ "Some important remarks when computing timings", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewBenchmark", [
        [ "What do the timings take into account?", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewTimings", null ],
        [ "Architecture options when compiling ProxSuite", "md_doc_22-PROXQP__API_22-ProxQP__api.html#OverviewArchitectureOptions", null ]
      ] ]
    ] ],
    [ "ProxQP solve function without API", "md_doc_23-ProxQP__solve.html", [
      [ "A single solve function for dense and sparse backends", "md_doc_23-ProxQP__solve.html#OverviewAsingleSolveFunction", null ]
    ] ],
    [ "Linear solvers API with examples", "md_doc_24-linearsolvers.html", [
      [ "Dense linear solver", "md_doc_24-linearsolvers.html#OverviewDense", null ],
      [ "Sparse linear solver", "md_doc_24-linearsolvers.html#OverviewSparse", null ]
    ] ],
    [ "Installation", "md_doc_25-installation.html", null ],
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
        [ "Related Symbols", "functions_rela.html", null ]
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
"classtl_1_1optional_3_01T_01_6_01_4.html#a71e275dea087ea60a60ee38b39b40d15",
"dense_2views_8hpp.html#a11de089956c90146573853cda2941473",
"get_8hpp.html#a6b5d3f5529533b36bc8d72a31336831e",
"macros_8hpp.html#a9087fb4c15212107761609645b9460f8",
"namespacemembers_func_t.html",
"namespaceproxsuite_1_1linalg_1_1veg.html#aae522d4ba2f2382dbadd8b7b7cadcc43ab736c00bb44fe7188d305cb2f3c72279",
"namespaceproxsuite_1_1linalg_1_1veg_1_1meta_1_1nb.html",
"namespaceproxsuite_1_1proxqp_1_1sparse.html#a03cc034874dbb17022d8996835154843",
"preprocessor_8hpp.html#a1ceb15d8b3dfbf43c9f2ec6ec96ba271",
"prologue_8hpp.html#a80928f93d0e43e3053fb76d4e83fcf03",
"sparse_2solver_8hpp.html#a1b12a295287ebefe88ccdfdb164f337c",
"structproxsuite_1_1linalg_1_1dense_1_1Ldlt.html#ab173d194d1463a12449df4d00655d344",
"structproxsuite_1_1linalg_1_1sparse_1_1MatMut.html#af26de7d15d3075a254b52f72979fe004",
"structproxsuite_1_1linalg_1_1veg_1_1Fix.html#a5c5f7843de36a59644207c53879f6aee",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1WithArg.html#a59c56c694c87a1bdb4edbf20d5eb8aef",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1__mem_1_1ManagedAlloc.html#a427fffafd8872a7c6d774448f1b32e7f",
"structproxsuite_1_1linalg_1_1veg_1_1__detail_1_1__meta_1_1zip__type__seq2_3_01F_00_01F_3_01Ts_8_296bf3865960d353995853a151ca4dd5.html#ae88be208f804d02ef921e5ee25d63c83",
"structproxsuite_1_1linalg_1_1veg_1_1dynstack_1_1DynStackArray.html#aa4a0ef2c98a38489f4a1531f47abcd8a",
"structproxsuite_1_1linalg_1_1veg_1_1meta_1_1nb_1_1is__consteval.html#a0d1a7ba1c216826748129a0f07a7ab27",
"structproxsuite_1_1proxqp_1_1Info.html#a7ec030095d08febf145c54d46f34d51f",
"structproxsuite_1_1proxqp_1_1StridedVectorViewMut.html#ac45febce90e76a0d11b179cd90afd2ce",
"structproxsuite_1_1proxqp_1_1dense_1_1QpViewMut.html",
"structproxsuite_1_1proxqp_1_1detail_1_1DetectedImpl_3_01Void_3_01F_3_01Ts_8_8_8_01_4_01_4_00_01F2ffd9abc2f7171836425854b294c9e6f.html#a59e4ed0c18691a02462bdf05239067a9",
"structproxsuite_1_1proxqp_1_1sparse_1_1SparseModel.html#a516b900ad0135214d3f9b327aba0e26b",
"structtl_1_1detail_1_1optional__copy__assign__base_3_01T_00_01false_01_4.html#a706147734aa595c6b575b1250a965bab",
"tl-optional_8hpp.html#a28faef14fa0efb12848f0cd4c087740e"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';