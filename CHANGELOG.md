# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Recursive stub generation for Python bindings ([#419](https://github.com/Simple-Robotics/proxsuite/pull/419))

### Changed
- Change the default branch to `devel` ([#395](https://github.com/Simple-Robotics/proxsuite/pull/395))
- Change `dual_feasibility` test threshold in `sparse_maros_meszaros` unit test ([#403](https://github.com/Simple-Robotics/proxsuite/pull/403))
- replace `std::numeric_limits<T>::infinity()` by `std::numeric_limits<T>::max()` ([#413](https://github.com/Simple-Robotics/proxsuite/pull/413))
- Upgraded nanobind dependency version (submodule) to v2.9.2 ([#418](https://github.com/Simple-Robotics/proxsuite/pull/418))
- Better dynamic module handling ([#419](https://github.com/Simple-Robotics/proxsuite/pull/419))

### Fixed
- Use the right table to store `configure-args` cmeel argument ([#403](https://github.com/Simple-Robotics/proxsuite/pull/403))
- Allow project to be used as an external CMake project (FetchContent) ([#408](https://github.com/Simple-Robotics/proxsuite/pull/408))
- Fix -Wdeprecated-literal-operator warning ([#420](https://github.com/Simple-Robotics/proxsuite/pull/420))

### Removed
- Don't release PyPy package on GNU/Linux anymore ([#403](https://github.com/Simple-Robotics/proxsuite/pull/403))

## [0.7.2] - 2025-03-12

### Fixed
* Fix an arcane compilation issue on Clang-19 ([#379](https://github.com/Simple-Robotics/proxsuite/pull/379))
* Replace `!= None` with `is not None` in Python bindings ([#375](https://github.com/Simple-Robotics/proxsuite/pull/375))
* Fix Internal compiler error with Visual Studio 17.13 ([#384](https://github.com/Simple-Robotics/proxsuite/pull/384))

### Changed
* Upgrade nanobind submodule to v2.5.0 ([#378](https://github.com/Simple-Robotics/proxsuite/pull/378))
* Switch to gersemi for formatting ([#380](https://github.com/Simple-Robotics/proxsuite/pull/380))
* Switch to ruff for formatting / linting ([#376](https://github.com/Simple-Robotics/proxsuite/pull/376))

## [0.7.1] - 2025-01-28

### Fixed
* Fix Windows build with MSVC compiler and C++20 standard ([#368](https://github.com/Simple-Robotics/proxsuite/pull/368))

## [0.7.0] - 2025-01-21

### Fixed
* CMake: Fix link to system cereal in tests ([#353](https://github.com/Simple-Robotics/proxsuite/pull/352))
* Fix windows build error related to template usage in veg ([#357](https://github.com/Simple-Robotics/proxsuite/pull/357))

### Added
* Stub files for Python bindings, using [nanobind's native support](https://nanobind.readthedocs.io/en/latest/typing.html#stub-generation) ([#340](https://github.com/Simple-Robotics/proxsuite/pull/340))
* Python 3.13 support on PyPI ([#361](https://github.com/Simple-Robotics/proxsuite/pull/361))
* Add `solve_no_gil` for dense backend (multithreading via python) ([#363](https://github.com/Simple-Robotics/proxsuite/pull/363))
* Add benchmarks for `solve_no_gil` vs `solve_in_parallel` (openmp) ([#363](https://github.com/Simple-Robotics/proxsuite/pull/363))

### Changed
* Change Python bindings to use nanobind instead of pybind11 ([#340](https://github.com/Simple-Robotics/proxsuite/pull/340))
* Update setup-minicondav2 to v3 ([#363](https://github.com/Simple-Robotics/proxsuite/pull/363))


## [0.6.7] - 2024-08-27

### Added
* Fix mu update function for PrimalLDLT backend ([#349](https://github.com/Simple-Robotics/proxsuite/pull/349))
* Allow use of installed pybind11, cereal and jrl-cmakemodules via cmake
* Add compatibility with jrl-cmakemodules workspace ([#339](https://github.com/Simple-Robotics/proxsuite/pull/339))
* Specifically mention that timings are in microseconds ([#342](https://github.com/Simple-Robotics/proxsuite/pull/342))
* Fix cereal include directory in cmake ([#342](https://github.com/Simple-Robotics/proxsuite/pull/342))
* Extend doc with hint for conda installation from source ([#342](https://github.com/Simple-Robotics/proxsuite/pull/342))

### Fixed
* Fix inequality constraints return in QPLayer ([#343](https://github.com/Simple-Robotics/proxsuite/pull/343))

### Changed

* Refactor Python examples with a new "util.py" file ([#347](https://github.com/Simple-Robotics/proxsuite/pull/347))

## [0.6.6] - 2024-06-15

### Fixed
* Fix infeasibility detection and add a unit test ([#328](https://github.com/Simple-Robotics/proxsuite/pull/328))

## [0.6.5] - 2024-05-31

### Added
* Pip wheels for Python 3.12 and stop support Python 3.7 ([#324](https://github.com/Simple-Robotics/proxsuite/pull/324))

### Fixed
* Fixes compilation issue with GCC 14 on Arch ([#322](https://github.com/Simple-Robotics/proxsuite/pull/322))

### What's Changed
* Change from torch.Tensor to torch.empty or torch.tensor and specify type explicitly ([#308](https://github.com/Simple-Robotics/proxsuite/pull/308))
* Fix handling of batch of inequality constraints in `QPFunctionFn_infeas`. The derivations in qplayer was done for single-sided constraints, that's the reason for the concatenation but the expansion of batchsize dimension was not working properly ([#308](https://github.com/Simple-Robotics/proxsuite/pull/308))
* Switch from self-hosted runner for macos-14-ARM to runner from github ([#306](https://github.com/Simple-Robotics/proxsuite/pull/306))
* Fix missing cassert for some compilers ([#316](https://github.com/Simple-Robotics/proxsuite/pull/316))

## [0.6.4] - 2024-03-01

### What's Changed
* Changed `primal_infeasibility_solving` to `False` for feasible QPs ([#302](https://github.com/Simple-Robotics/proxsuite/pull/302))

## [0.6.3] - 2024-01-23

### Fixed
* Fix Python tests with scipy>=1.12 ([#299](https://github.com/Simple-Robotics/proxsuite/pull/299))

## [0.6.2] - 2024-01-22

### Fixed
* Fix Windows build ([#290](https://github.com/Simple-Robotics/proxsuite/pull/290))
* Fix math formulae in documentation ([#294](https://github.com/Simple-Robotics/proxsuite/pull/294))
* Restore correc values for infeasibility ([#292](https://github.com/Simple-Robotics/proxsuite/pull/292))
* Handles CPU/GPU transfer in `QPFunctionFn`'s `backward` function ([#297](https://github.com/Simple-Robotics/proxsuite/pull/297))


## [0.6.1] - 2023-11-16

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#280](https://github.com/Simple-Robotics/proxsuite/pull/280))
* Templating power iteration algorithm by matrix storage order by [@quentinll](https://github.com/quentinll) ([#279](https://github.com/Simple-Robotics/proxsuite/pull/279))

### New Contributors
* [@quentinll](https://github.com/quentinll) made their first contribution ([#279](https://github.com/Simple-Robotics/proxsuite/pull/279))


## [0.6.0] - 2023-11-13

### News
We add the implementation of [QPLayer](https://inria.hal.science/hal-04133055/file/QPLayer_Preprint.pdf).
**QPLayer** enables to use a QP as a layer within standard learning architectures.
**QPLayer** allows for parallelized calculus over CPUs, and is interfaced with **PyTorch**.
**QPLayer** can also differentiate over LPs.

### What's Changed
* QPLayer: efficient differentiation of convex quadratic optimization by [@fabinsch,](https://github.com/fabinsch,) [@Bambade](https://github.com/Bambade) and [@quentinll](https://github.com/quentinll) ([#264](https://github.com/Simple-Robotics/proxsuite/pull/264))


## [0.5.1] - 2023-11-09

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#265](https://github.com/Simple-Robotics/proxsuite/pull/265))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#268](https://github.com/Simple-Robotics/proxsuite/pull/268))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#269](https://github.com/Simple-Robotics/proxsuite/pull/269))
* Check model is_valid up to eps by [@fabinsch](https://github.com/fabinsch) ([#272](https://github.com/Simple-Robotics/proxsuite/pull/272))


## [0.5.0] - 2023-09-26

This release adds [**support for nonconvex QPs**](https://github.com/Simple-Robotics/proxsuite/issues/237), along with healthy fixes.

### What's Changed
* Fix compilation (veg/memory) for gcc 7 and clang 7 by [@costashatz](https://github.com/costashatz) ([#255](https://github.com/Simple-Robotics/proxsuite/pull/255))
* Estimate minimal eigenvalue of quadratic cost hessian by [@Bambade](https://github.com/Bambade) ([#257](https://github.com/Simple-Robotics/proxsuite/pull/257))
* Fix typo #254  by [@Bambade](https://github.com/Bambade) ([#258](https://github.com/Simple-Robotics/proxsuite/pull/258))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#260](https://github.com/Simple-Robotics/proxsuite/pull/260))
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) ([#261](https://github.com/Simple-Robotics/proxsuite/pull/261))

### New Contributors
* [@costashatz](https://github.com/costashatz) made their first contribution ([#255](https://github.com/Simple-Robotics/proxsuite/pull/255))


## [0.4.1] - 2023-08-02

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#247](https://github.com/Simple-Robotics/proxsuite/pull/247))
* Add Iros on ROS CI by [@jcarpent](https://github.com/jcarpent) ([#248](https://github.com/Simple-Robotics/proxsuite/pull/248))
* Update default value for update_preconditioner by [@jcarpent](https://github.com/jcarpent) ([#250](https://github.com/Simple-Robotics/proxsuite/pull/250))


## [0.4.0] - 2023-07-10

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#225](https://github.com/Simple-Robotics/proxsuite/pull/225))
* add generalized primal dual augmented Lagrangian (gpdal) for dense backend by [@Bambade](https://github.com/Bambade) ([#228](https://github.com/Simple-Robotics/proxsuite/pull/228))
* optimize dense iterative refinement by [@Bambade](https://github.com/Bambade) ([#230](https://github.com/Simple-Robotics/proxsuite/pull/230))
* Add dense LP interface by [@Bambade](https://github.com/Bambade) ([#231](https://github.com/Simple-Robotics/proxsuite/pull/231))
* Enable solving QP ([#229](parallel with ProxQP  by [@Bambade](https://github.com/Bambade) in https://github.com/Simple-Robotics/proxsuite/pull/229))
* Add small dense LP Python example by [@stephane-caron](https://github.com/stephane-caron) ([#235](https://github.com/Simple-Robotics/proxsuite/pull/235))
* Fix typo ([#234](dense lp interface by [@Bambade](https://github.com/Bambade) in https://github.com/Simple-Robotics/proxsuite/pull/234))
* ci check all jobs pass by [@fabinsch](https://github.com/fabinsch) ([#236](https://github.com/Simple-Robotics/proxsuite/pull/236))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#232](https://github.com/Simple-Robotics/proxsuite/pull/232))
* Add box constraint interface for dense backend by [@Bambade](https://github.com/Bambade) ([#238](https://github.com/Simple-Robotics/proxsuite/pull/238))
* Improve dense backend and simplify calculus when using a Diagonal Hessian by [@Bambade](https://github.com/Bambade) ([#239](https://github.com/Simple-Robotics/proxsuite/pull/239))
* Add infeasibility solving feature for dense and sparse backends by [@Bambade](https://github.com/Bambade) ([#241](https://github.com/Simple-Robotics/proxsuite/pull/241))
* cmake: fix path to find-external/OpenMP by [@fabinsch](https://github.com/fabinsch) ([#240](https://github.com/Simple-Robotics/proxsuite/pull/240))
* More information ([#242](debug mode by [@fabinsch](https://github.com/fabinsch) in https://github.com/Simple-Robotics/proxsuite/pull/242))
* Fix warning and clean // solvers API by [@jcarpent](https://github.com/jcarpent) ([#243](https://github.com/Simple-Robotics/proxsuite/pull/243))


## [0.3.7] - 2023-05-05

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#206](https://github.com/Simple-Robotics/proxsuite/pull/206))
* Define PROXSUITE_AS_SUBPROJECT as ON by [@amiller27](https://github.com/amiller27) ([#207](https://github.com/Simple-Robotics/proxsuite/pull/207))
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) ([#208](https://github.com/Simple-Robotics/proxsuite/pull/208))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#210](https://github.com/Simple-Robotics/proxsuite/pull/210))
* Set default CMAKE_BUILD_TYPE value by [@jcarpent](https://github.com/jcarpent) ([#211](https://github.com/Simple-Robotics/proxsuite/pull/211))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#214](https://github.com/Simple-Robotics/proxsuite/pull/214))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#215](https://github.com/Simple-Robotics/proxsuite/pull/215))
* Set simde dependency for ROS2 Iron as well by [@wxmerkt](https://github.com/wxmerkt) ([#218](https://github.com/Simple-Robotics/proxsuite/pull/218))
* Sync submodule CMake by [@jcarpent](https://github.com/jcarpent) ([#219](https://github.com/Simple-Robotics/proxsuite/pull/219))

### New Contributors
* [@amiller27](https://github.com/amiller27) made their first contribution ([#207](https://github.com/Simple-Robotics/proxsuite/pull/207))


## [0.3.6] - 2023-03-14

### What's Changed
* ci/linux-wheels: fix naming for artifacts by [@fabinsch](https://github.com/fabinsch) ([#198](https://github.com/Simple-Robotics/proxsuite/pull/198))
* pip wheels for aarch64 by [@fabinsch](https://github.com/fabinsch) ([#202](https://github.com/Simple-Robotics/proxsuite/pull/202))


## [0.3.5] - 2023-03-06

### What's Changed
* doc: pip install available on windows by [@fabinsch](https://github.com/fabinsch) ([#189](https://github.com/Simple-Robotics/proxsuite/pull/189))
* set compute_timings = true by [@fabinsch](https://github.com/fabinsch) ([#191](https://github.com/Simple-Robotics/proxsuite/pull/191))
* [minor] Clean up invalid link from pyproject.toml by [@stephane-caron](https://github.com/stephane-caron) ([#193](https://github.com/Simple-Robotics/proxsuite/pull/193))
* ci/linux: fix linux wheel compatibility by [@fabinsch](https://github.com/fabinsch) ([#196](https://github.com/Simple-Robotics/proxsuite/pull/196))


## [0.3.4] - 2023-03-01

### What's Changed
* CI: pip wheels on windows by [@fabinsch](https://github.com/fabinsch) ([#185](https://github.com/Simple-Robotics/proxsuite/pull/185))
* [CI] ROS: Add friendly names; add TSID on Rolling as downstream test by [@wxmerkt](https://github.com/wxmerkt) ([#184](https://github.com/Simple-Robotics/proxsuite/pull/184))
* CI: simplify workflow on self-hosted M1 by [@fabinsch](https://github.com/fabinsch) ([#187](https://github.com/Simple-Robotics/proxsuite/pull/187))


## [0.3.3] - 2023-02-25

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#172](https://github.com/Simple-Robotics/proxsuite/pull/172))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#174](https://github.com/Simple-Robotics/proxsuite/pull/174))
* cmake: add option BUILD_DOCUMENTATION by [@fabinsch](https://github.com/fabinsch) ([#177](https://github.com/Simple-Robotics/proxsuite/pull/177))
* linalg/ldlt : mark p() and pt() permutation matrix getters const by [@ManifoldFR](https://github.com/ManifoldFR) ([#180](https://github.com/Simple-Robotics/proxsuite/pull/180))
* feat: enable simde ([#21](package.xml by [@wep21))](https://github.com/wep21))) ([#178](https://github.com/Simple-Robotics/proxsuite/pull/178))
* [CI] ROS: Activate prerelease by [@wxmerkt](https://github.com/wxmerkt) ([#181](https://github.com/Simple-Robotics/proxsuite/pull/181))

### New Contributors
* [@ManifoldFR](https://github.com/ManifoldFR) made their first contribution ([#180](https://github.com/Simple-Robotics/proxsuite/pull/180))
* [@wep21](https://github.com/wep21) made their first contribution ([#178](https://github.com/Simple-Robotics/proxsuite/pull/178))


## [0.3.2] - 2023-01-17

### What's Changed
* Expose check_duality_gap ([#167](Python settings by [@stephane-caron](https://github.com/stephane-caron) in https://github.com/Simple-Robotics/proxsuite/pull/167))
* Add duality-gap thresholds by [@stephane-caron](https://github.com/stephane-caron) ([#169](https://github.com/Simple-Robotics/proxsuite/pull/169))

### New Contributors
* [@stephane-caron](https://github.com/stephane-caron) made their first contribution ([#167](https://github.com/Simple-Robotics/proxsuite/pull/167))


## [0.3.1] - 2023-01-09

### What's Changed
* Fix package.xml and add ROS-CI by [@wxmerkt](https://github.com/wxmerkt) ([#158](https://github.com/Simple-Robotics/proxsuite/pull/158))
* ROS-CI: Debug and Release builds, fix warnings by [@wxmerkt](https://github.com/wxmerkt) ([#159](https://github.com/Simple-Robotics/proxsuite/pull/159))
* Fix computation of duality gap quantity by [@jcarpent](https://github.com/jcarpent) ([#163](https://github.com/Simple-Robotics/proxsuite/pull/163))
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) ([#164](https://github.com/Simple-Robotics/proxsuite/pull/164))

### New Contributors
* [@wxmerkt](https://github.com/wxmerkt) made their first contribution ([#158](https://github.com/Simple-Robotics/proxsuite/pull/158))


## [0.3.0] - 2022-12-26

### What's Changed
* Add serialization of dense qp model using cereal by [@fabinsch](https://github.com/fabinsch) ([#152](https://github.com/Simple-Robotics/proxsuite/pull/152))
* transposeInPlace only if eigen >= 3.4.0 by [@fabinsch](https://github.com/fabinsch) ([#153](https://github.com/Simple-Robotics/proxsuite/pull/153))
* Fix Windows C++20 compatibility by [@jcarpent](https://github.com/jcarpent) ([#155](https://github.com/Simple-Robotics/proxsuite/pull/155))


## [0.2.16] - 2022-12-21

### What's Changed
* Fix default parameter values for compute_timings by [@jcarpent](https://github.com/jcarpent) ([#146](https://github.com/Simple-Robotics/proxsuite/pull/146))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#147](https://github.com/Simple-Robotics/proxsuite/pull/147))
* Fix packaging issue on Windows by [@jcarpent](https://github.com/jcarpent) ([#149](https://github.com/Simple-Robotics/proxsuite/pull/149))


## [0.2.15] - 2022-12-15

### What's Changed
* cmake/python: update library output paths by [@fabinsch](https://github.com/fabinsch) ([#144](https://github.com/Simple-Robotics/proxsuite/pull/144))


## [0.2.14] - 2022-12-14

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#134](https://github.com/Simple-Robotics/proxsuite/pull/134))
* update benchmarks random mixed qp by [@fabinsch](https://github.com/fabinsch) ([#137](https://github.com/Simple-Robotics/proxsuite/pull/137))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#138](https://github.com/Simple-Robotics/proxsuite/pull/138))
* add is_valid function for dense model and fix example + unittest by [@fabinsch](https://github.com/fabinsch) ([#139](https://github.com/Simple-Robotics/proxsuite/pull/139))


## [0.2.13] - 2022-11-29

### What's Changed
* Enforce check for other architectures by [@jcarpent](https://github.com/jcarpent) ([#132](https://github.com/Simple-Robotics/proxsuite/pull/132))
* Fix support for POWERPC by [@jcarpent](https://github.com/jcarpent) ([#133](https://github.com/Simple-Robotics/proxsuite/pull/133))


## [0.2.12] - 2022-11-26

### What's Changed
* Fix compilation with mingw by [@jgillis](https://github.com/jgillis) ([#130](https://github.com/Simple-Robotics/proxsuite/pull/130))

### New Contributors
* [@jgillis](https://github.com/jgillis) made their first contribution ([#130](https://github.com/Simple-Robotics/proxsuite/pull/130))


## [0.2.11] - 2022-11-25

### What's Changed
* Fix dimension check issues by [@jcarpent](https://github.com/jcarpent) ([#125](https://github.com/Simple-Robotics/proxsuite/pull/125))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#124](https://github.com/Simple-Robotics/proxsuite/pull/124))
* Add is_initialized safeguard to workspace by [@fabinsch](https://github.com/fabinsch) ([#126](https://github.com/Simple-Robotics/proxsuite/pull/126))
* Add simple cpp example by [@fabinsch](https://github.com/fabinsch) ([#127](https://github.com/Simple-Robotics/proxsuite/pull/127))


## [0.2.10] - 2022-11-17

### What's Changed
* Allow macOS Debug pipeline to fail -> checking for memory alloc by [@fabinsch](https://github.com/fabinsch) ([#117](https://github.com/Simple-Robotics/proxsuite/pull/117))
* Devel by [@jcarpent](https://github.com/jcarpent) ([#118](https://github.com/Simple-Robotics/proxsuite/pull/118))
* fix instructionset import by [@fabinsch](https://github.com/fabinsch) ([#120](https://github.com/Simple-Robotics/proxsuite/pull/120))


## [0.2.9] - 2022-11-14

### What's Changed
* Enforce robustness computation of duality gap by [@jcarpent](https://github.com/jcarpent) ([#114](https://github.com/Simple-Robotics/proxsuite/pull/114))
* Assert primal/dual residual and dual gap != nan + add unittest by [@fabinsch](https://github.com/fabinsch) ([#115](https://github.com/Simple-Robotics/proxsuite/pull/115))


## [0.2.8] - 2022-11-12

### What's Changed
* Enforce constness and clean CMake by [@jcarpent](https://github.com/jcarpent) ([#106](https://github.com/Simple-Robotics/proxsuite/pull/106))
* Add details on using ProxSuite with CMake projects by [@jcarpent](https://github.com/jcarpent) ([#111](https://github.com/Simple-Robotics/proxsuite/pull/111))
* Add duality gap measure by [@Bambade](https://github.com/Bambade) ([#110](https://github.com/Simple-Robotics/proxsuite/pull/110))


## [0.2.7] - 2022-11-10

### What's Changed
* CMakeLists: fix INTERFACE target compile definitions by [@fabinsch](https://github.com/fabinsch) ([#97](https://github.com/Simple-Robotics/proxsuite/pull/97))
* sync submodule by [@fabinsch](https://github.com/fabinsch) ([#98](https://github.com/Simple-Robotics/proxsuite/pull/98))
* Fixed temporaries ([#100](update() of dense wrapper by [@fennel-labs](https://github.com/fennel-labs) in https://github.com/Simple-Robotics/proxsuite/pull/100))
* Additional option to check eigen runtime malloc by [@fabinsch](https://github.com/fabinsch) ([#93](https://github.com/Simple-Robotics/proxsuite/pull/93))

### New Contributors
* [@fennel-labs](https://github.com/fennel-labs) made their first contribution ([#100](https://github.com/Simple-Robotics/proxsuite/pull/100))


## [0.2.6] - 2022-11-08

### What's Changed
* User option sparse backend by [@fabinsch](https://github.com/fabinsch) ([#84](https://github.com/Simple-Robotics/proxsuite/pull/84))
* CI for c++14 compatibility by [@fabinsch](https://github.com/fabinsch) ([#88](https://github.com/Simple-Robotics/proxsuite/pull/88))
* Update README.md by [@fabinsch](https://github.com/fabinsch) ([#89](https://github.com/Simple-Robotics/proxsuite/pull/89))
* CI: use v1.2 also for windows, add cxx_std to key by [@fabinsch](https://github.com/fabinsch) ([#90](https://github.com/Simple-Robotics/proxsuite/pull/90))
* Fix heap allocation ([#92](dense solver by [@jcarpent](https://github.com/jcarpent) in https://github.com/Simple-Robotics/proxsuite/pull/92))


## [0.2.5] - 2022-11-06

### What's Changed
* C++14 compliant implementation of optional by [@fabinsch](https://github.com/fabinsch) ([#78](https://github.com/Simple-Robotics/proxsuite/pull/78))
* C++14 compliant implementation of aligned_alloc by [@fabinsch](https://github.com/fabinsch) ([#79](https://github.com/Simple-Robotics/proxsuite/pull/79))
* unittest/sparse-ruiz: replace checks with isApprox by [@fabinsch](https://github.com/fabinsch) ([#83](https://github.com/Simple-Robotics/proxsuite/pull/83))
* Fix packaging for pip by [@jcarpent](https://github.com/jcarpent) ([#82](https://github.com/Simple-Robotics/proxsuite/pull/82))
* Move optional.hpp to the right place by [@jcarpent](https://github.com/jcarpent) ([#81](https://github.com/Simple-Robotics/proxsuite/pull/81))
* Fix logic and bug ([#85](warm_start by [@jcarpent](https://github.com/jcarpent) in https://github.com/Simple-Robotics/proxsuite/pull/85))


## [0.2.4] - 2022-11-01

### What's Changed
* Change eps_abs to a reasonable value by [@jcarpent](https://github.com/jcarpent) ([#70](https://github.com/Simple-Robotics/proxsuite/pull/70))
* sparse/solver: use refactorize for matrix free solver by [@fabinsch](https://github.com/fabinsch) ([#73](https://github.com/Simple-Robotics/proxsuite/pull/73))
* Enhance API  by [@jcarpent](https://github.com/jcarpent) ([#71](https://github.com/Simple-Robotics/proxsuite/pull/71))
* Correct documentation of bindings by [@fabinsch](https://github.com/fabinsch) ([#72](https://github.com/Simple-Robotics/proxsuite/pull/72))
* Add pipeline with MSVC compiler by [@fabinsch](https://github.com/fabinsch) ([#49](https://github.com/Simple-Robotics/proxsuite/pull/49))


## [0.2.3] - 2022-10-29

### What's Changed
* Fix epsilon relative stopping criteria by [@fabinsch](https://github.com/fabinsch) ([#58](https://github.com/Simple-Robotics/proxsuite/pull/58))
* Handle empty inputs by [@fabinsch](https://github.com/fabinsch) ([#59](https://github.com/Simple-Robotics/proxsuite/pull/59))
* Fix calculation of objection function value by [@fabinsch](https://github.com/fabinsch) ([#60](https://github.com/Simple-Robotics/proxsuite/pull/60))
* Doc: matio link by [@fabinsch](https://github.com/fabinsch) ([#61](https://github.com/Simple-Robotics/proxsuite/pull/61))
* Fix packaging issue by [@jcarpent](https://github.com/jcarpent) ([#64](https://github.com/Simple-Robotics/proxsuite/pull/64))
* Change default model upper and lower bounds  by [@Bambade](https://github.com/Bambade) ([#66](https://github.com/Simple-Robotics/proxsuite/pull/66))
* Fix API for bounds on the inequality constraints by [@jcarpent](https://github.com/jcarpent) ([#67](https://github.com/Simple-Robotics/proxsuite/pull/67))


## [0.2.2] - 2022-10-19

### What's Changed
* release: wheels for macos arm64 by [@fabinsch](https://github.com/fabinsch) ([#53](https://github.com/Simple-Robotics/proxsuite/pull/53))
* Don't compile AVX Python bindings on non X86_64 arch by [@jcarpent](https://github.com/jcarpent) ([#54](https://github.com/Simple-Robotics/proxsuite/pull/54))
* Fix existence of std::aligned_alloc on APPLE by [@jcarpent](https://github.com/jcarpent) ([#55](https://github.com/Simple-Robotics/proxsuite/pull/55))


## [0.2.1] - 2022-10-18

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#43](https://github.com/Simple-Robotics/proxsuite/pull/43))
* Sync ma([#42](with devel by [@jcarpent](https://github.com/jcarpent) in https://github.com/Simple-Robotics/proxsuite/pull/42))
* Fix compile options export for Windows by [@jcarpent](https://github.com/jcarpent) ([#44](https://github.com/Simple-Robotics/proxsuite/pull/44))
* benchmark: document speed up by vectorization by [@fabinsch](https://github.com/fabinsch) ([#48](https://github.com/Simple-Robotics/proxsuite/pull/48))
* Support ARM64 architecture by [@fabinsch](https://github.com/fabinsch) ([#50](https://github.com/Simple-Robotics/proxsuite/pull/50))


## [0.2.0] - 2022-10-08

This release introduces a notable change ([#](the order of bounds constraints.))
As the API is not yet totally fixed, we have only increased the minor release version.

More to come ([#](a forthcoming release.))

### What's Changed
* CI/release: fix uploading proxsuite wheels only by [@fabinsch](https://github.com/fabinsch) ([#30](https://github.com/Simple-Robotics/proxsuite/pull/30))
* TEST-AUTOMERGE by [@fabinsch](https://github.com/fabinsch) ([#31](https://github.com/Simple-Robotics/proxsuite/pull/31))
* Fix c++ documentation by [@fabinsch](https://github.com/fabinsch) ([#32](https://github.com/Simple-Robotics/proxsuite/pull/32))
* Compilation MSVC by [@aescande](https://github.com/aescande) ([#33](https://github.com/Simple-Robotics/proxsuite/pull/33))
* Add specific compile option for MSVC by [@jcarpent](https://github.com/jcarpent) ([#35](https://github.com/Simple-Robotics/proxsuite/pull/35))
* Sync submodule CMake by [@jcarpent](https://github.com/jcarpent) ([#37](https://github.com/Simple-Robotics/proxsuite/pull/37))
* Remove sparse sparse overload for dense backend by [@Bambade](https://github.com/Bambade) ([#36](https://github.com/Simple-Robotics/proxsuite/pull/36))
* change qp.init(H,g,A,b,C,u,l) into qp.init(H,g,A,b,C,l,u) by [@Bambade](https://github.com/Bambade) ([#39](https://github.com/Simple-Robotics/proxsuite/pull/39))
* Fix packaging issues and remove useless test files by [@jcarpent](https://github.com/jcarpent) ([#40](https://github.com/Simple-Robotics/proxsuite/pull/40))

### New Contributors
* [@aescande](https://github.com/aescande) made their first contribution ([#33](https://github.com/Simple-Robotics/proxsuite/pull/33))


## [0.1.2] - 2022-09-26

### What's Changed
* Support python 3.7 for proxsuite wheels by [@fabinsch](https://github.com/fabinsch) ([#27](https://github.com/Simple-Robotics/proxsuite/pull/27))
* Add default proximal step sizes by [@Bambade](https://github.com/Bambade) ([#26](https://github.com/Simple-Robotics/proxsuite/pull/26))
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) ([#28](https://github.com/Simple-Robotics/proxsuite/pull/28))


## [0.1.1] - 2022-09-09

### What's Changed
* add missing includes by [@nim65s](https://github.com/nim65s) ([#21](https://github.com/Simple-Robotics/proxsuite/pull/21))
* update README and sync submodule by [@Bambade](https://github.com/Bambade) ([#20](https://github.com/Simple-Robotics/proxsuite/pull/20))
* Fix packaging issues by [@jcarpent](https://github.com/jcarpent) ([#23](https://github.com/Simple-Robotics/proxsuite/pull/23))
* Fix preconditioner by [@fabinsch](https://github.com/fabinsch) ([#24](https://github.com/Simple-Robotics/proxsuite/pull/24))
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) ([#25](https://github.com/Simple-Robotics/proxsuite/pull/25))
* Add more unittests by [@fabinsch](https://github.com/fabinsch) ([#22](https://github.com/Simple-Robotics/proxsuite/pull/22))

### New Contributors
* [@nim65s](https://github.com/nim65s) made their first contribution ([#21](https://github.com/Simple-Robotics/proxsuite/pull/21))
* [@pre-commit-ci](https://github.com/pre-commit-ci) made their first contribution ([#25](https://github.com/Simple-Robotics/proxsuite/pull/25))


## [0.1.0] - 2022-08-24

### What's Changed
* Fix aligned alloc for old version of OSX target deployment by [@jcarpent](https://github.com/jcarpent) ([#3](https://github.com/Simple-Robotics/proxsuite/pull/3))
* Fix documentation and the publishing pipeline by [@jcarpent](https://github.com/jcarpent) ([#4](https://github.com/Simple-Robotics/proxsuite/pull/4))
* Bindings: expose qp solver output by [@fabinsch](https://github.com/fabinsch) ([#11](https://github.com/Simple-Robotics/proxsuite/pull/11))
* provide initialization of solvers with None entries  by [@Bambade](https://github.com/Bambade) ([#12](https://github.com/Simple-Robotics/proxsuite/pull/12))
* Fix packaging issues and add more packaging test by [@jcarpent](https://github.com/jcarpent) ([#17](https://github.com/Simple-Robotics/proxsuite/pull/17))
* Use PROXSUITE_VECTORIZE and change logic by [@jcarpent](https://github.com/jcarpent) ([#18](https://github.com/Simple-Robotics/proxsuite/pull/18))

### New Contributors
* [@jcarpent](https://github.com/jcarpent) made their first contribution ([#3](https://github.com/Simple-Robotics/proxsuite/pull/3))
* [@fabinsch](https://github.com/fabinsch) made their first contribution ([#11](https://github.com/Simple-Robotics/proxsuite/pull/11))
* [@Bambade](https://github.com/Bambade) made their first contribution ([#12](https://github.com/Simple-Robotics/proxsuite/pull/12))


## [0.0.1] - 2022-08-11

The first release of ProxSuite.


[Unreleased]: https://github.com/Simple-Robotics/proxsuite/compare/v0.7.2...HEAD
[0.7.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.7...v0.7.0
[0.6.7]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.7...v0.4.0
[0.3.7]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.3...v0.3.4
[0.3.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.16...v0.3.0
[0.2.16]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.15...v0.2.16
[0.2.15]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.14...v0.2.15
[0.2.14]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.3...v0.2.4
[0.2.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/Simple-Robotics/proxsuite/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Simple-Robotics/proxsuite/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Simple-Robotics/proxsuite/compare/v0.0.1...v0.1.0
