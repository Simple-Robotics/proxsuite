#ifndef INRIA_LDLT_TEST_ADD_ROW_SOLVER_HPP_HDWGZKCLS
#define INRIA_LDLT_TEST_ADD_ROW_SOLVER_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include <ldlt/ldlt.hpp>
#include "qp/views.hpp"
#include "ldlt/factorize.hpp"
#include "ldlt/detail/meta.hpp"
#include "ldlt/solve.hpp"
#include "ldlt/update.hpp"
#include "qp/line_search.hpp"
#include <cmath>

namespace qp {
inline namespace tags {
using namespace ldlt::tags;
}

namespace detail {

template <typename T>
void insertRowTest() {

	using namespace ldlt::tags;

	isize dim = 10;
	
	LDLT_MULTI_WORKSPACE_MEMORY(
			(_m_init,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T)
			(_m_to_get_,Uninit, Mat(dim, dim),LDLT_CACHELINE_BYTES, T),
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

    row<< T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8),T(9),T(10),T(11) ;

    ldl.insert_at(dim, row);

    for (isize i = 0; i < dim; ++i) {
		m_to_get(i, i) += T(1);
	}	
    for (isize i = 0 ; i < dim+1 ; ++i){
        m_to_get(dim, i) = T(i+1) ;
    }

    std::cout << " diff " << m_to_get -  ldl.reconstructed_matrix() << std::endl;
    std::cout << "m_to_get " << m_to_get << std::endl;

}


} // namespace detail

} // namespace qp

#endif /* end of include guard INRIA_LDLT_TEST_ADD_ROW_SOLVER_HPP_HDWGZKCLS */
