#include <numeric>
#include <matioCpp/matioCpp.h>

using Scalar = double;

auto main() -> int {
	using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	using ldlt::i64;
	using ldlt::isize;

    matioCpp::File input("/home/antoine/Bureau/thèse/projects/prox-qp/solver_f/solver_biding_test/maros_meszaros_data/AUG2DC.mat");
    
}
 