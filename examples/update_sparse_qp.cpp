#include "qp/sparse/sparse.hpp" // get the sparse API of ProxQP
#include "test/include/util.hpp" // use a function for generating a random QP

using namespace qp;
using T = double;
using I = c_int;
int main()
{
	I n = 10;
	I n_eq(n / 4);
	I n_in(n / 4);
    T p = 0.15; // level of sparsity
    T conditioning = 10.0; // conditioning level for H
    auto H = ldlt_test::rand::sparse_positive_definite_rand(n, conditioning, p);
    auto g = ldlt_test::rand::vector_rand<T>(n);
    auto A = ldlt_test::rand::sparse_matrix_rand<T>(n_eq,n, p);
    auto C = ldlt_test::rand::sparse_matrix_rand<T>(n_in,n, p);
    auto x_sol = ldlt_test::rand::vector_rand<T>(n);
    auto b = A * x_sol;
    auto l =  C * x_sol; 
    auto u = (l.array() + 10).matrix().eval();
    // design a Qp2 object using sparsity masks of H, A and C
    qp::sparse::QP<T,I> Qp(H.cast<bool>(),A.cast<bool>(),C.cast<bool>());
    Qp.init(H,g,A,b,C,u,l);
    Qp.solve();
    // update H 
    auto H_new = 2* H; // keep the same sparsity structure
    Qp.update(H_new,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt); // update H with H_new, it will work
    Qp.solve();
    // generate H2 with another sparsity structure
    auto H2 = ldlt_test::rand::sparse_positive_definite_rand(n, conditioning, p);
    Qp.update(H2,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt) ; // nothing will happen
    // if only a vector changes, then the update takes effect
    auto g_new = ldlt_test::rand::vector_rand<T>(n); 
    Qp.update(std::nullopt,g,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt);
    Qp.solve(); // it solves the problem with another vector
    // to solve the problem with H2 matrix create a new Qp object
    qp::sparse::QP<T,I> Qp2(H2.cast<bool>(),A.cast<bool>(),C.cast<bool>());
    Qp2.init(H2,g_new,A,b,C,u,l);
    Qp2.solve();// it will solve the new problem
}
