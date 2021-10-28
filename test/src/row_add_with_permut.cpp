#include <Eigen/Core>
#include <doctest.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <util.hpp>
#include <ldlt/ldlt.hpp>
#include "ldlt/views.hpp"
#include <ldlt/update.hpp>
#include <ldlt/factorize.hpp>

using namespace ldlt;
using T = f64;

DOCTEST_TEST_CASE("row add") {

    using namespace ldlt::tags;

	isize dim = 10;
	
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_m_init,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T),
			(_m_to_get_,Uninit, Mat(dim+1, dim+1),LDLT_CACHELINE_BYTES, T),
            (row_,Init, Vec(dim+1),LDLT_CACHELINE_BYTES, T)
		);

	auto m_init = _m_init.to_eigen();
    auto m_to_get = _m_to_get_.to_eigen();
    auto row = row_.to_eigen();

    m_init.setZero();
    m_to_get.setZero();
    row.setZero();
	for (isize i = 0; i < dim; ++i) {
		m_init(i, i) += T(1);
	}
	
	ldlt::Ldlt<T> ldl{decompose, m_init};

    std::cout << "initial difference " << m_init - ldl.reconstructed_matrix() << std::endl;

    row<< 1,2,3,4,5,6,7,8,9,10,11 ;

    std::cout << "initial row " << row << std::endl;
    
    ldl.insert_at(dim, row);

    for (isize i = 0; i < dim; ++i) {
		m_to_get(i, i) += T(1);
	}	

    std::cout << "m_to_get " << m_to_get << std::endl;
    for (isize i = 0 ; i < dim+1 ; ++i){
        m_to_get(dim, i) = T(i+1) ;
        m_to_get(i, dim) = T(i+1) ;
    }
    std::cout << "m_to_get " << m_to_get << std::endl;

    std::cout << " diff " << m_to_get -  ldl.reconstructed_matrix() << std::endl;
    std::cout << "m_to_get " << m_to_get << std::endl;
	T eps = T(1e-10);
	DOCTEST_CHECK((m_to_get - ldl.reconstructed_matrix()).norm() <= eps);
}
