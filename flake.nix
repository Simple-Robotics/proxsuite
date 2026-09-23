{
  description = "Advanced Proximal Optimization Toolbox";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        extraDevPyPackages = [ "proxsuite" ];
        overrideAttrs.proxsuite = { drv-prev, pkgs-final, ... }: {
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./benchmark
              ./bindings
              ./cmake
              ./CMakeLists.txt
              ./doc
              ./examples
              ./include
              ./package.xml
              ./test
            ];
          };
          patches = [ ];
          cmakeFlags = drv-prev.cmakeFlags ++ [
            (lib.cmakeBool "BUILD_TESTING" true)
            (lib.cmakeBool "PROXSUITE_BUILD_EXAMPLES" true)
            (lib.cmakeBool "BUILD_BENCHMARK" true)
            (lib.cmakeBool "BUILD_WITH_OPENMP_SUPPORT" true)
            (lib.cmakeBool "PROXSUITE_BUILD_MAROS_MESZAROS_TESTS" true)
            (lib.cmakeBool "GENERATE_PYTHON_STUBS" true)
          ];
          buildInputs =
            drv-prev.buildInputs
            ++ lib.optionals pkgs-final.stdenv.cc.isClang [
              pkgs-final.llvmPackages.openmp
            ];
          checkInputs = drv-prev.checkInputs ++ [
            pkgs-final.catch2_3
          ];
        };
        extends.eigen5 = final: prev: {
          eigen = final.eigen_5;
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (_python-final: python-prev: {
              scipy = python-prev.scipy.overrideAttrs {
                # broken on linux arm
                doInstallCheck = false;
              };
            })
          ];
        };
      }
    );
}
