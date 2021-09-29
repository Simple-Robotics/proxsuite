#ifndef INRIA_LDLT_LINE_SEARCH_HPP_HDWGZKCLS
#define INRIA_LDLT_LINE_SEARCH_HPP_HDWGZKCLS

#include "ldlt/views.hpp"
#include "qp/views.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <list>

namespace qp {

namespace line_search {

template <typename Scalar> 
struct line_search_result {
    Scalar grad; 
    Scalar a0 ; 
    Scalar b0 ;
};

struct active_set_change_result {
    Eigen::Matrix<i32, Eigen::Dynamic, 1> new_bijection_map; 
    i32 n_c_f ;
};


template <typename Scalar,Layout LC>
auto gradient_norm_computation( //
        VectorView<Scalar> ze,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dz,
        Scalar& mu_eq,
        Scalar& mu_in,
        Scalar& rho,
		MatrixView<Scalar,LC> C,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Cdx,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_primal_residual_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& primal_residual_eq,
        Scalar& alpha,
        i32 dim, 
        i32 n_eq,
        i32 n_in
        ) -> Scalar {

    /* Compute the squared norm of the following vector
    H(x+alpha dx) + rho * (x+alpha dx - xe) + A.T * (y+alpha dy ) + C(active).T * (z+alpha dz)_+ 
    A*(x+alpha dx) - b - (y+alpha dy - ye)/mu_eq 
    C(active)*(x+alpha dx) - d(active) - ( (z_acti+alpha dz_active)_+ - z_e(active) ) / mu_in 
    (z_inactive + alpha dz_inactive)_+ 
    */

    //auto C_copy = qp::detail::to_eigen_matrix(C);
    auto C_copy = C.to_eigen();
	//auto z_e = qp::detail::to_eigen_vector(ze);
    auto z_e = ze.to_eigen();

    // define active set
    auto tmp = residual_in_z + alpha * Cdx;
    auto active_set_tmp = (tmp).array() >= 0 ;  // /!\ error lacks Cdx * alpha
    auto num_active = active_set_tmp.count();
    auto num_inactive = n_in-num_active;
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set(num_active);
    Eigen::Matrix<i32, Eigen::Dynamic, 1> inactive_set(num_inactive);
    i32 i = 0;
    i32 j = 0;
    for (i32 k =0; k<n_in;++k){
        if (active_set_tmp(k)) {
            active_set(i) = k;
            ++i;
        }
        else{
            inactive_set(j)= k;
            ++j;
        }
    }

    // form the gradient
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> positive_part_z =  ( (z_e + alpha * dz).array() > 0).select(z_e + alpha * dz, Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(n_in));

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res(dim+n_eq+n_in); 
    
    res.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq ;

    for (i32 k =0; k<num_active;++k){
        res.topRows(dim)  += positive_part_z(active_set(k)) * C_copy.row(active_set(k)) ; 
    }
    //std::cout << " --- num_active " << num_active << " active set "  << active_set << std::endl << std::endl ; 
    //std::cout << " --- positive part " << positive_part_z << std::endl << std::endl; 
    //std::cout << " --- dual " << res.topRows(dim).squaredNorm() << std::endl << std::endl ; 
    res.middleRows(dim, n_eq) = primal_residual_eq + alpha * d_primal_residual_eq ; 

    for (i32 k =0; k<num_active;++k){
        res(dim+n_eq+k) = tmp(active_set(k)) - positive_part_z(active_set(k)) / mu_in   ; 
        
    }
    //std::cout << " --- primal equality " << res.middleRows(dim, n_eq).squaredNorm() << std::endl << std::endl ; 

    for (i32 k =0; k<num_inactive;++k){
        res(dim+n_eq+num_active+k) = positive_part_z(inactive_set(k)) ; 
    }

    //std::cout << " --- primal inequality " << res.tail(num_active).squaredNorm() << std::endl << std::endl ; 

    // return the squared norm
    return res.squaredNorm() ;

}

template <typename Scalar,Layout LC>
auto gradient_norm_computation_box( //
        VectorView<Scalar> ze,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dz,
        Scalar& mu_eq,
        Scalar& mu_in,
        Scalar& rho,
		MatrixView<Scalar,LC> C,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Cdx,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z_u,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z_l,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_primal_residual_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& primal_residual_eq,
        Scalar& alpha,
        i32 dim, 
        i32 n_eq,
        i32 n_in
        ) -> Scalar {

    /* Compute the squared norm of the following vector res

    vect1 = H.dot(x) + g + rho_primal_proximal * (x-xe) + A.transpose().dot(y) +  C[active_inequalities_l,:].transpose().dot(z[active_inequalities_l]) + C[active_inequalities_u,:].transpose().dot(z[active_inequalities_u])
    vect3_u = (residual_in_u[active_inequalities_u] * mu_in  - (z[active_inequalities_u]-ze[active_inequalities_u]))/mu_in
    vect3_l = (residual_in_l[active_inequalities_l] * mu_in - (z[active_inequalities_l]-ze[active_inequalities_l]))/mu_in
    vect4 = z[inactive_inequalities]
    res = np.concatenate( (vect1, (residual_eq * mu_eq - (y-ye) )/mu_eq , vect3_u , vect3_l, vect4), axis = None )

    cosidering the following qp problem : (H, g, A, b, C, u,l) and

    residual_eq = A.dot(x) - b
    residual_in_u = C.dot(x) - u
    residual_in_l = C.dot(x) - l
    active_inequalities_u = residual_in_u + z/mu_in >= 0
    active_inequalities_u = residual_in_l + z/mu_in <= 0
    active_inequalities = active_inequalities_u + active_inequalities_l
    inactive_inequalities = ~active_inequalities  

    */

    //auto C_copy = qp::detail::to_eigen_matrix(C);
    auto C_copy =  C.to_eigen();
	//auto z_e = qp::detail::to_eigen_vector(ze);
    auto z_e = ze.to_eigen();

    // define active set
    auto tmp_u = residual_in_z_u+alpha * Cdx;
    auto tmp_l = residual_in_z_l+alpha * Cdx;
    //auto active_set_tmp_u = (tmp_u).array() >= 0 ; /!\ give bugs
    //auto active_set_tmp_l = (tmp_l).array() <= 0 ; 
    //auto inactive_set_tmp = ((tmp_u).array() < 0).array() && ((tmp_l).array() > 0).array();
    //auto num_inactive = inactive_set_tmp.count() ; 
    //auto num_active_u = active_set_tmp_u.count();
    //auto num_active_l = active_set_tmp_l.count();

   
    i32 num_active_u = 0 ; 
    i32 num_active_l = 0 ; 
    i32 num_inactive = 0 ;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k)>=Scalar(0.)) {
            num_active_u += 1;
            std::cout << "active_u tmp_u(k) " << tmp_u(k) << std::endl;
        }
        if (tmp_l(k)<=Scalar(0.)) {
            num_active_l += 1;
            std::cout << "active_l tmp_l(k) " << tmp_l(k) << std::endl;
        }
        if (tmp_u(k) <Scalar(0.) && tmp_l(k)>Scalar(0.)) {
            num_inactive += 1;
            std::cout << "inactive tmp_u(k) " << tmp_u(k) << " inactive tmp_l(k) " << tmp_l(k) << std::endl;
        }
    }
    std::cout <<  "num_inactive " << num_inactive << " num_active_u " << num_active_u <<  " num_active_l " << num_active_l <<" sum " << num_active_u + num_active_l+num_inactive << " n_in " << n_in << std::endl;
    //std::cout <<  "num_inactive_ " << num_inactive_ << " num_active_u_ " << num_active_u_ <<  " num_active_l_ " << num_active_l_ << std::endl;
    
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_u(num_active_u);
    active_set_u.setZero();
    i32 i_u = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k)>=Scalar(0.)) {
            //std::cout << "k " << k << " tmp_u(k) " << tmp_u(k) << " i_u " << i_u << std::endl;
            active_set_u(i_u) = k;
            i_u+=1;
        }
    }
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_l(num_active_l);
    active_set_l.setZero();
    i32 i_l = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_l(k)<=Scalar(0.)) {
            //std::cout << "k " << k << " tmp_l(k) " << tmp_u(k) << " i_l " << i_l << std::endl;
            active_set_l(i_l) = k;
            i_l+=1;
        }
    }

    Eigen::Matrix<i32, Eigen::Dynamic, 1> inactive_set(num_inactive);
    i32 i_inact = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k) <Scalar(0.) && tmp_l(k)>Scalar(0.)) {
            inactive_set(i_inact) = k;
            i_inact+=1;
        }
    }

    // form the gradient
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> active_part_z(n_in) ; 
    
    active_part_z = z_e + alpha * dz ;
    //std::cout << "Ok here " << std::endl;
    //std::cout << "active_set_u " << active_set_u << std::endl;
    //std::cout << "active_part_z before " << active_part_z << std::endl;
    for (i32 k =0; k<num_active_u;k=k+1){
        if ( active_part_z(active_set_u(k)) < Scalar(0.)   ){
            //std::cout << "k " <<  k << " active_set_u(k) "<< active_set_u(k) << " n_in " << n_in <<std::endl;
            active_part_z(active_set_u(k)) =  Scalar(0.);
        } 
    }
    //std::cout << "Ok here z u" << std::endl;
    //std::cout << "active_set_l " << active_set_l << std::endl;
    for (i32 k =0; k<num_active_l;k=k+1){

        if (active_part_z(active_set_l(k)) > Scalar(0.)) {
            //std::cout << "k " <<  k << " active_set_l(k) "<< active_set_l(k) << " n_in " << n_in <<std::endl;
            active_part_z(active_set_l(k)) = Scalar(0.);
        }
    }
    //std::cout << "active_part_z after " << active_part_z << std::endl;
    //std::cout << "Ok here z l" << std::endl;
    //std::cout << "active_part_z : " << active_part_z << std::endl;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res(dim+n_eq+n_in); 
    res.setZero();
    
    res.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq ;
    //std::cout << "res.topRows(dim) inactive norm " << (res.topRows(dim)).squaredNorm() << std::endl;
    //std::cout << "Ok here as well" << std::endl;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> aux_u(dim);
    aux_u.setZero();
    for (i32 k =0; k<num_active_u;++k){
        res.topRows(dim)  += active_part_z(active_set_u(k)) * C_copy.row(active_set_u(k)) ; 
        aux_u +=active_part_z(active_set_u(k)) * C_copy.row(active_set_u(k)) ;
    }
    //std::cout << "res.topRows(dim) active u norm " << (aux_u).squaredNorm() << std::endl;
    //std::cout << "Ok here for top rows u" << std::endl;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> aux_l(dim);
    aux_l.setZero();
    for (i32 k =0; k<num_active_l;++k){
        res.topRows(dim)  += active_part_z(active_set_l(k)) * C_copy.row(active_set_l(k)) ; 
        aux_l += active_part_z(active_set_l(k)) * C_copy.row(active_set_l(k)) ;
    }
    //std::cout << "res.topRows(dim) active l norm " << (aux_l).squaredNorm() << std::endl;
    //std::cout << "res.topRows(dim) norm " << (res.topRows(dim)).squaredNorm() << std::endl;
    //std::cout << "Ok here for top rows l" << std::endl;
    res.middleRows(dim, n_eq) = primal_residual_eq + alpha * d_primal_residual_eq ; 
    //std::cout << "res.middleRows(dim, n_eq) norm" << (res.middleRows(dim, n_eq)).squaredNorm() << std::endl;
    //std::cout << "Ok here for middle rows eq" << std::endl;
    for (i32 k =0; k<num_active_u;++k){
        res(dim+n_eq+k) = tmp_u(active_set_u(k)) - active_part_z(active_set_u(k)) / mu_in   ; 
    }
    //std::cout << "res.middleRows(dim+n_eq, num_active_u) norm" << (res.middleRows(dim+n_eq, num_active_u)).squaredNorm() << std::endl;
    //std::cout << "Ok here for tail rows u" << std::endl;
    for (i32 k =0; k<num_active_l;++k){
        res(dim+n_eq+num_active_u+k) = tmp_l(active_set_l(k)) - active_part_z(active_set_l(k)) / mu_in   ; 
        
    }
    //std::cout << "res.middleRows(dim+n_eq+num_active_u, num_num_active_lactive_u) norm" << (res.middleRows(dim+n_eq+num_active_u, num_active_l)).squaredNorm() << std::endl;
    //std::cout << "Ok here for top tail row l" << std::endl;
    for (i32 k =0; k<num_inactive;++k){
        res(dim+n_eq+num_active_u+num_active_l+k) = active_part_z(inactive_set(k)) ; 
    }
    //std::cout << "res.tail(num_inactive) norm" << (res.tail(num_inactive)).squaredNorm() << std::endl;
    //std::cout << "Ok here for tail row inactive" << std::endl;
    // return the squared norm
    return res.squaredNorm() ;

}

template <typename Scalar>
auto gradient_norm_qpalm( //
		VectorView<Scalar> x,
        VectorView<Scalar> xe,
        VectorView<Scalar> dx,
        Scalar& mu_eq,
        Scalar& mu_in,
        Scalar& rho,
        Scalar alpha,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Hdx, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&   g, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Adx, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  residual_in_y, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  residual_in_z, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Cdx,
        i32 n_in) ->  Scalar {

    /* Compute the squared norm of the following vector

    residual_in = C.dot(x+alpha*dx) - d
    
    active_set = residual_in * mu_in + ze > 0.

    a0 = 0.5 * dx.transpose().dot(H.dot(dx)) + (mu_eq/2.) * norm(A.dot(dx))**2 + (mu_in/2.) * norm(C[active_set,:].dot(dx))**2 + (rho/(2.)) * norm(dx)**2
    
    b0 = x.transpose().dot(H.dot(dx)) + (rho) * dx.transpose().dot(x-xe) +  g.transpose().dot(dx) + mu_eq * (A.dot(dx)).transpose().dot(A.dot(x)-b+ye/(mu_eq)) + mu_in * (C[active_set,:].dot(dx)).transpose().dot( (C.dot(x)-d+(ze/mu_in))[active_set] ) 

    grad = 2.* a0 * alpha + b0 

    return grad, a0, b0
    */


	//auto x_ = qp::detail::to_eigen_vector(x);
    //auto xe_ = qp::detail::to_eigen_vector(xe);
    //auto dx_ = qp::detail::to_eigen_vector(dx);

	auto x_ = x.to_eigen();
    auto xe_ = xe.to_eigen();
    auto dx_ = dx.to_eigen();

    // define active set

    auto active_set_tmp =  (residual_in_z + Cdx * alpha).array() > 0   ; 
    
    auto num_active = active_set_tmp.count();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> active_set(num_active);
    i32 i = 0;
    for (i32 k =0; k<n_in;++k){
        if (active_set_tmp(k)) {
            active_set(i) = k;
            ++i;
        }
    }

    // coefficient computation

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_a0(num_active);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_b0(num_active);

    for (i32 k =0; k<num_active;++k){
        tmp_a0(k) = Cdx(active_set(k)) ; 
        tmp_b0(k) = residual_in_z(active_set(k)) ; 
    }

    Scalar a = dx_.dot(Hdx)  + mu_eq * (Adx).squaredNorm() + mu_in * tmp_a0.squaredNorm() + rho * dx_.squaredNorm() ;
    Scalar b = x_.dot(Hdx ) + (rho * (x_-xe_) + g).dot(dx_) + mu_eq * (Adx).dot(residual_in_y) +mu_in * tmp_a0.dot(tmp_b0) ; 

    return a * alpha + b ;
}


template <typename Scalar>
auto gradient_norm_qpalm_box( //
		VectorView<Scalar> x,
        VectorView<Scalar> xe,
        VectorView<Scalar> dx,
        Scalar& mu_eq,
        Scalar& mu_in,
        Scalar& rho,
        Scalar alpha,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Hdx, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&   g, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Adx, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  residual_in_y, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  residual_in_z_u, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  residual_in_z_l, 
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&  Cdx,
        i32 n_in) ->  Scalar {

    /* 
    
    the function computes the first derivative of the proximal augmented lagrangian of the problem 
    
    phi(alpha) = f(x_k+alpha dx) + rho /2 ||x_k + alpha dx - x_k||**2 + mu_eq/2 ( ||A(x_k+alpha dx)-d+y_k/mu_eq||**2 - ||y_k/mu_eq||**2 ) + mu_in/2 ( || [C(x_k+alpha dx) - u + z_k/mu_in]_+ ||**2 +  || [C(x_k+alpha dx) - l + z_k/mu_in]_- ||**2  - ||z_k / mu_in ||**2 ) 
    with f(x) = 0.5 * x^THx + g^Tx

    phi is a second order polynomial in alpha. Below are computed its coefficient a0 and b0 in order to compute the desired gradient a0 * alpha + b0

    */

	auto x_ = x.to_eigen();
    auto xe_ = xe.to_eigen();
    auto dx_ = dx.to_eigen();

    // define active set
    auto tmp_u = residual_in_z_u + Cdx * alpha;
    auto tmp_l = residual_in_z_l + Cdx * alpha;
    //std::cout << "tmp_l " << tmp_l << std::endl;
    auto active_set_tmp_u =  (tmp_u).array() > 0   ; 
    auto active_set_tmp_l =  (tmp_l).array() < 0   ; 

    i32 num_active_u = 0 ; 
    i32 num_active_l = 0 ; 
    for (i32 k =0; k<n_in;k=k+1){
        if (active_set_tmp_u(k)) {
            num_active_u += 1;
        }
        if (active_set_tmp_l(k)) {
            num_active_l += 1;
        }
    }

    //auto num_active_u = active_set_tmp_u.count(); // /!\ big with this function, returns not the exact number of active sets...
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_u(num_active_u);
    active_set_u.setZero();
    //auto num_active_l = active_set_tmp_l.count();
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_l(num_active_l);
    active_set_l.setZero();
    i32 i = 0;
    i32 j = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (active_set_tmp_u(k)) {
            active_set_u(i) = k;
            i+=1;
        }
        if (active_set_tmp_l(k)) {
            //std::cout << "tmp_l(k) " << tmp_l(k) << std::endl; 
            active_set_l(j) = k;
            j+=1;
        }
    }

    // coefficient computation

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_a0_u(num_active_u);
    tmp_a0_u.setZero();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_b0_u(num_active_u);
    tmp_b0_u.setZero();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_a0_l(num_active_l);
    tmp_a0_l.setZero();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_b0_l(num_active_l);
    tmp_b0_l.setZero();

    for (i32 k =0; k<num_active_u;k=k+1){
        tmp_a0_u(k) = Cdx(active_set_u(k)) ; 
        tmp_b0_u(k) = residual_in_z_u(active_set_u(k)) ; 
    }
    for (i32 k =0; k<num_active_l;k=k+1){
        tmp_a0_l(k) = Cdx(active_set_l(k)) ; 
        tmp_b0_l(k) = residual_in_z_l(active_set_l(k)) ; 
    }
    //std::cout << "num_active_l "<< num_active_l << std::endl;
    for (i32 k =0; k<num_active_l;k=k+1){
        //std::cout << "k " << k  << " active_set_l(k) " <<active_set_l(k)<< " tmp_l(k) " << tmp_l(active_set_l(k)) << " residual_in_z_l(active_set_l(k)) " <<  residual_in_z_l(active_set_l(k)) << std::endl; 
        tmp_a0_l(k) = Cdx(active_set_l(k)) ; 
        tmp_b0_l(k) = residual_in_z_l(active_set_l(k)) ; 
    }    

    Scalar a_no_act = dx_.dot(Hdx)  + mu_eq * (Adx).squaredNorm()  + rho * dx_.squaredNorm() ;
    Scalar a_act = mu_in * (tmp_a0_u.squaredNorm()+tmp_a0_l.squaredNorm()) ; 

    Scalar a = dx_.dot(Hdx)  + mu_eq * (Adx).squaredNorm() + mu_in * (tmp_a0_u.squaredNorm()+tmp_a0_l.squaredNorm()) + rho * dx_.squaredNorm() ;

    Scalar b_no_act = x_.dot(Hdx ) + (rho * (x_-xe_) + g).dot(dx_) + mu_eq * (Adx).dot(residual_in_y) ; 
    Scalar b_act = mu_in * (tmp_a0_l.dot(tmp_b0_l) + tmp_a0_u.dot(tmp_b0_u) ); 
    Scalar b_act_l =  mu_in * (tmp_a0_l.dot(tmp_b0_l)) ; 
    Scalar b_act_u =  mu_in * (tmp_a0_u.dot(tmp_b0_u)) ; 

    Scalar b = x_.dot(Hdx ) + (rho * (x_-xe_) + g).dot(dx_) + mu_eq * (Adx).dot(residual_in_y) +mu_in * (tmp_a0_l.dot(tmp_b0_l) + tmp_a0_u.dot(tmp_b0_u) ); 

    
    //std::cout << " a_no_act " << a_no_act << " a_act " << a_act << " b_no_act " << b_no_act << " b_act " << b_act << " b_act_l " << b_act_l <<  " b_act_u " << b_act_u  << std::endl;
    std::cout << "a0 " <<  a << " b0 " << b << " alpha " << alpha << " grad " << a * alpha + b << std::endl;
    //std::cout << "residual_in_z_l[active_set_l] "<< tmp_b0_l << " Cdx(active_set_l(k)) " << tmp_a0_l << " mu_in "<< mu_in << std::endl;
    
    return a * alpha + b ;
}


template <
		typename Scalar,
		Layout LC>
auto local_saddle_point( //
        VectorView<Scalar> ze,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dz_,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		MatrixView<Scalar, LC> C,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Cdx,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_primal_residual_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& primal_residual_eq,
        Scalar& alpha,
        i32& dim, 
        i32& n_eq,
        i32& n_in
        ) -> Scalar {
    /*
    the function returns the local saddle point wrt the L2 norm of the gradient of the proximal augmented lagrangian for the corresponding active set.
    The corresponding argmin alpha of this loss function is modified inplace. 

    polynomial coefficients are computed with this exact algebra using 

                z = ze.copy()
                dz = dz.copy()
                z[z+a*dz<0.] = 0.
                dz[z+a*dz<0.] = 0.

                active_set = C.dot(x+a*dx)-d + ze/mu_in >= 0.

    H_ = H + sparse.csc_matrix(rho*np.identity(n)) 

    a0 = norm( H_.dot(dx) +  A.transpose().dot(dy) + C[active_inequalities,:].transpose().dot(dz[active_inequalities])  )**2 + norm( (mu_eq * A.dot(dx) - dy)/mu_eq )**2 + norm( (mu_in * C[active_inequalities,:].dot(dx) - dz[active_inequalities])/mu_in )**2    + norm( dz[inactive_inequalities])**2
    
    b0 = 2. * ( H_.dot(dx) + A.transpose().dot(dy)  + C[active_inequalities,:].transpose().dot(dz[active_inequalities]) ).dot( H.dot(x) + rho*(x-xe)+ g + A.transpose().dot(y) + C[active_inequalities,:].transpose().dot(z[active_inequalities]) )
    b0 += 2. * ( residual_eq - (y-ye)/mu_eq ).dot( (mu_eq * A.dot(dx) - dy)/mu_eq ) + 2. * (residual_in[active_inequalities] - (z[active_inequalities]-ze[active_inequalities])/mu_in ).dot(  (mu_in * C[active_inequalities,:].dot(dx) - dz[active_inequalities])/mu_in)    + 2. * (z[inactive_inequalities]).dot( dz[inactive_inequalities])

    c0 = norm( H.dot(x) +rho*(x-xe)+ g + A.transpose().dot(y) + C[active_inequalities,:].transpose().dot(z[active_inequalities]) )**2  + norm(residual_eq- (y-ye)/mu_eq )**2 + norm(residual_in[active_inequalities] - (z[active_inequalities]-ze[active_inequalities])/mu_in )**2    + norm(z[inactive_inequalities])**2 

    */
	//auto C_copy = qp::detail::to_eigen_matrix(C);
	//auto z_e = qp::detail::to_eigen_vector(ze);
	auto C_copy = C.to_eigen();
	auto z_e = ze.to_eigen();

    // compute positive parts of z and dz
    auto positive_part = ( z_e + alpha * dz_ ).array() > 0 ; 
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> z_p =  ( positive_part).select(z_e, Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(n_in));
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> dz_p =  ( positive_part).select(dz_, Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(n_in));

    // define active set
    auto active_set_tmp = ((residual_in_z + alpha * Cdx).array() >= 0) ; 
    auto num_active = active_set_tmp.count();
    auto num_inactive = n_in-num_active;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> active_set(num_active);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> inactive_set(num_inactive);
    i32 i = 0;
    i32 j = 0;
    for (i32 k =0; k<n_in;++k){
        if (active_set_tmp(k)) {
            active_set(i) = k;
            ++i;
        }
        else{
            inactive_set(j)= k;
            ++j;
        }
    }

    // a0 computation 
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_d2(num_active);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_d3(num_inactive);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp2(num_active);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp3(num_inactive);

    for (i32 k =0; k<num_active;++k){
        d_dual_for_eq += dz_p(active_set(k)) * C_copy.row(active_set(k)) ;
        tmp_d2(k) =  Cdx(active_set(k))  - dz_p(active_set(k)) / mu_in ; 
    }
    for (i32 k =0; k<num_inactive;++k){
        tmp_d3(k) = dz_p(inactive_set(k)) ; 
    }
    Scalar a0 = d_dual_for_eq.squaredNorm() + tmp_d2.squaredNorm() + tmp_d3.squaredNorm() + d_primal_residual_eq.squaredNorm() ; 

    // b0 computation

    for (i32 k =0; k<num_active;++k){
        dual_for_eq += z_p(active_set(k)) * C_copy.row(active_set(k)) ;
        tmp2(k) =  residual_in_z(active_set(k)) - z_p(active_set(k))/ mu_in   ; 
    }
    for (i32 k =0; k<num_inactive;++k){
        tmp3(k) = z_p(inactive_set(k)) ; 
    }

    Scalar b0 = 2 * ( d_dual_for_eq.dot(dual_for_eq) + tmp_d2.dot(tmp2) +  tmp3.dot(tmp_d3)  + primal_residual_eq.dot(d_primal_residual_eq) ) ; 

    // c0 computation
    Scalar c0 = dual_for_eq.squaredNorm() + tmp2.squaredNorm() + tmp3.squaredNorm() + primal_residual_eq.squaredNorm() ; 

    // derivation of the loss function value and corresponding argmin alpha

    auto res = Scalar(0);
    
    if (a0!=0){
        alpha = (-b0/(2*a0)) ; 
        res = a0 * pow(alpha, Scalar(2)) + b0 * alpha + c0 ; 
    } else if (b0!= 0){
        alpha = (-c0/(b0)) ; 
        res = b0 * alpha + c0;
    }
    else{
        alpha = 0 ; 
        res = c0;
    }
    
    //// unit test : check the value is the same as the squared norm of the gradient
    // cannot be derived with other function as positive part with given alpha, is not the same as the one used for deriving the gradient norm
    
    /*
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> positive_part_z =  (positive_part).select(z_e + alpha * dz_, Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(n_in));

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_unit_test(dim+n_eq+n_in); 
    
    tmp_unit_test.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq;

    std::cout << "norm diff for dual part " << tmp_unit_test.topRows(dim).squaredNorm()  - d_dual_for_eq.squaredNorm() * pow(alpha, Scalar(2)) - 2*  d_dual_for_eq.dot(dual_for_eq) * alpha - dual_for_eq.squaredNorm()<< std::endl << std::endl ;

    tmp_unit_test.middleRows(dim, n_eq) = primal_residual_eq + alpha *d_primal_residual_eq ; 

    std::cout << "norm diff for equality part " << tmp_unit_test.middleRows(dim, n_eq).squaredNorm()  - d_primal_residual_eq.squaredNorm() * pow(alpha, Scalar(2)) - 2*  d_primal_residual_eq.dot(primal_residual_eq) * alpha - primal_residual_eq.squaredNorm()<< std::endl << std::endl ;

    for (i32 k =0; k<num_active;++k){
        tmp_unit_test(dim+n_eq+k) =  residual_in_z(active_set(k)) + (Cdx(active_set(k))*alpha  - (positive_part_z(active_set(k)) )/ mu_in) ; 
    }
    

    for (i32 k =0; k<num_inactive;++k){
        tmp_unit_test(dim+n_eq+num_active+k) = positive_part_z(inactive_set(k)) ; 
    }

    std::cout << "norm diff for active inequality part " << tmp_unit_test.middleRows(dim+n_eq,num_active).squaredNorm()  - tmp_d2.squaredNorm() * pow(alpha, Scalar(2)) - 2*  tmp_d2.dot(tmp2) * alpha - tmp2.squaredNorm()<< std::endl << std::endl ;
    std::cout << "norm diff for inactive inequality part " << tmp_unit_test.middleRows(dim+n_eq+num_active,num_inactive).squaredNorm()  - tmp_d3.squaredNorm() * pow(alpha, Scalar(2)) - 2*  tmp_d3.dot(tmp3) * alpha - tmp3.squaredNorm()<< std::endl << std::endl ;

    std::cout << "norm diff " << tmp_unit_test.squaredNorm()  - res << std::endl << std::endl ;
    */

	return res;
}

template <
		typename Scalar,
		Layout LC>
auto local_saddle_point_box( //
        VectorView<Scalar> ze,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& dz_,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		MatrixView<Scalar, LC> C,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& Cdx,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z_u,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& residual_in_z_l,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_for_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& d_primal_residual_eq,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& primal_residual_eq,
        Scalar& alpha,
        i32& dim, 
        i32& n_eq,
        i32& n_in
        ) -> Scalar {
    /*
    the function returns the unique minimum of the positive second order polynomial in alpha of the L2 norm of the following vector

    H.dot(x) + g + rho_primal_proximal * (x-xe) + A.transpose().dot(y) +  C[active_inequalities_l,:].transpose().dot(z[active_inequalities_l]) + C[active_inequalities_u,:].transpose().dot(z[active_inequalities_u])
    (residual_eq * mu_eq - (y-ye) )/mu_eq
    (residual_in_u[active_inequalities_u] * mu_in  - (z[active_inequalities_u]-ze[active_inequalities_u]))/mu_in
    (residual_in_l[active_inequalities_l] * mu_in - (z[active_inequalities_l]-ze[active_inequalities_l]))/mu_in
    z[inactive_inequalities]

    with 
        x = xe + alpha dx ; y = ye + alpha dy ; z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u],0) and z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l],0)
    
    Furthermore 
        residual_eq = A.dot(x) - b
        residual_in_u = C.dot(x) - u
        residual_in_l = C.dot(x) - l
        active_inequalities_u = residual_in_u + alpha Cdx >=0
        active_inequalities_l = residual_in_l + alpha Cdx <=0
        active_inequalities = active_inequalities_u + active_inequalities_l
        inactive_inequalities = ~active_inequalities  
    
    To do so the L2 norm is expanded and the exact coefficients of the polynomial a0 alpha**2 + b0 alpha + c0 are derived.
    The argmin is then equal to -b0/2a0 if a0 != 0 and is changed INPLACE (erasing then alpha entry)

    the function returns the L2 norm of the merit function evaluated at the argmin value found
    
    */
	//auto C_copy = qp::detail::to_eigen_matrix(C);
	//auto z_e = qp::detail::to_eigen_vector(ze);
    auto tmp_u = residual_in_z_u+alpha*Cdx ;
    auto tmp_l = residual_in_z_l+alpha*Cdx ;
	auto C_copy = C.to_eigen();
	auto z_e = ze.to_eigen();

    i32 num_active_u = 0 ; 
    i32 num_active_l = 0 ; 
    i32 num_inactive = 0 ;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k)>=Scalar(0.)) {
            num_active_u += 1;
            //std::cout << "active_u tmp_u(k) " << tmp_u(k) << std::endl;
        }
        if (tmp_l(k)<=Scalar(0.)) {
            num_active_l += 1;
            //std::cout << "active_l tmp_l(k) " << tmp_l(k) << std::endl;
        }
        if (tmp_u(k) <Scalar(0.) && tmp_l(k)>Scalar(0.)) {
            num_inactive += 1;
            //std::cout << "inactive tmp_u(k) " << tmp_u(k) << " inactive tmp_l(k) " << tmp_l(k) << std::endl;
        }
    }
    //std::cout <<  "num_inactive " << num_inactive << " num_active_u " << num_active_u <<  " num_active_l " << num_active_l <<" sum " << num_active_u + num_active_l+num_inactive << " n_in " << n_in << std::endl;
    //std::cout <<  "num_inactive_ " << num_inactive_ << " num_active_u_ " << num_active_u_ <<  " num_active_l_ " << num_active_l_ << std::endl;
    
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_u(num_active_u);
    active_set_u.setZero();
    i32 i_u = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k)>=Scalar(0.)) {
            //std::cout << "k " << k << " tmp_u(k) " << tmp_u(k) << " i_u " << i_u << std::endl;
            active_set_u(i_u) = k;
            i_u+=1;
        }
    }
    Eigen::Matrix<i32, Eigen::Dynamic, 1> active_set_l(num_active_l);
    active_set_l.setZero();
    i32 i_l = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_l(k)<=Scalar(0.)) {
            //std::cout << "k " << k << " tmp_l(k) " << tmp_u(k) << " i_l " << i_l << std::endl;
            active_set_l(i_l) = k;
            i_l+=1;
        }
    }

    Eigen::Matrix<i32, Eigen::Dynamic, 1> inactive_set(num_inactive);
    i32 i_inact = 0;
    for (i32 k =0; k<n_in;k=k+1){
        if (tmp_u(k) <Scalar(0.) && tmp_l(k)>Scalar(0.)) {
            inactive_set(i_inact) = k;
            i_inact+=1;
        }
    }

    // form the gradient
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> z_p(n_in) ; 
    z_p = z_e ;
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> dz_p(n_in) ; 
    dz_p =  dz_ ; 

    for (i32 k =0; k<num_active_u;++k){
        Scalar test = z_e(active_set_u(k))  + alpha * dz_(active_set_u(k));
        if ( test < 0   ){
            z_p(active_set_u(k)) =  0;
            dz_p(active_set_u(k)) =  0;
        } 
    }
    for (i32 k =0; k<num_active_l;++k){
        Scalar test2 = z_e(active_set_l(k))  + alpha * dz_(active_set_l(k));
        if (test2 > 0) {
            z_p(active_set_l(k)) =  0;
            dz_p(active_set_l(k)) =  0;
        }
    }
    /*
    a0 = norm( H_.dot(dx) +  A.transpose().dot(dy) + C[active_inequalities_u,:].transpose().dot(dz[active_inequalities_u]) + C[active_inequalities_l,:].transpose().dot(dz[active_inequalities_l])  )**2 + norm( (mu_eq * A.dot(dx) - dy)/mu_eq )**2 
    a0 += norm( (mu_in * C[active_inequalities_u,:].dot(dx) - dz[active_inequalities_u])/mu_in )**2   + norm( (mu_in * C[active_inequalities_l,:].dot(dx) - dz[active_inequalities_l])/mu_in )**2  +  norm( dz[inactive_inequalities])**2
    
    b0 = 2. * ( H_.dot(dx) + A.transpose().dot(dy)  + C[active_inequalities_u,:].transpose().dot(dz[active_inequalities_u]) + C[active_inequalities_l,:].transpose().dot(dz[active_inequalities_l]) ).dot( H.dot(x) + rho*(x-xe)+ g + A.transpose().dot(y) + C[active_inequalities_u,:].transpose().dot(z[active_inequalities_u]) + C[active_inequalities_l,:].transpose().dot(z[active_inequalities_l]) )
    b0 += 2. * ( residual_eq - (y-ye)/mu_eq ).dot( (mu_eq * A.dot(dx) - dy)/mu_eq ) + 2. * (residual_in_u[active_inequalities_u] - (z[active_inequalities_u]-ze[active_inequalities_u])/mu_in ).dot(  (mu_in * C[active_inequalities_u,:].dot(dx) - dz[active_inequalities_u])/mu_in)    + 2. * (z[inactive_inequalities]).dot( dz[inactive_inequalities])
    b0 += + 2. * (residual_in_l[active_inequalities_l] - (z[active_inequalities_l]-ze[active_inequalities_l])/mu_in ).dot(  (mu_in * C[active_inequalities_l,:].dot(dx) - dz[active_inequalities_l])/mu_in)
    c0 = norm( H.dot(x) +rho*(x-xe)+ g + A.transpose().dot(y) + C[active_inequalities_u,:].transpose().dot(z[active_inequalities_u])  + C[active_inequalities_l,:].transpose().dot(z[active_inequalities_l]) )**2  + norm(residual_eq- (y-ye)/mu_eq )**2 
    c0 += norm(residual_in_u[active_inequalities_u] - (z[active_inequalities_u]-ze[active_inequalities_u])/mu_in )**2 + norm(residual_in_l[active_inequalities_l] - (z[active_inequalities_l]-ze[active_inequalities_l])/mu_in )**2  + norm(z[inactive_inequalities])**2 
    */

    // a0 computation 
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_d2_u(num_active_u);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_d2_l(num_active_l);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_d3(num_inactive);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp2_u(num_active_u);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp2_l(num_active_l);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp3(num_inactive);

    for (i32 k =0; k<num_active_u;++k){
        d_dual_for_eq += dz_p(active_set_u(k)) * C_copy.row(active_set_u(k)) ;
        tmp_d2_u(k) =  Cdx(active_set_u(k))  - dz_p(active_set_u(k)) / mu_in ; 
    }
    for (i32 k =0; k<num_active_l;++k){
        d_dual_for_eq += dz_p(active_set_l(k)) * C_copy.row(active_set_l(k)) ;
        tmp_d2_l(k) =  Cdx(active_set_l(k))  - dz_p(active_set_l(k)) / mu_in ; 
    }

    for (i32 k =0; k<num_inactive;++k){
        tmp_d3(k) = dz_p(inactive_set(k)) ; 
    }
    //Scalar a0 = d_dual_for_eq.squaredNorm() + tmp_d2_u.squaredNorm() + tmp_d3_u.squaredNorm() + tmp_d2_l.squaredNorm() + tmp_d3_l.squaredNorm() + d_primal_residual_eq.squaredNorm() ; 
    Scalar a0 = d_dual_for_eq.squaredNorm() + tmp_d2_u.squaredNorm() + tmp_d2_l.squaredNorm() + tmp_d3.squaredNorm() + d_primal_residual_eq.squaredNorm() ; 
    //std::cout << "a0 " << a0 << std::endl;
    // b0 computation

    for (i32 k =0; k<num_active_u;++k){
        dual_for_eq += z_p(active_set_u(k)) * C_copy.row(active_set_u(k)) ;
        tmp2_u(k) =  residual_in_z_u(active_set_u(k)) - z_p(active_set_u(k))/ mu_in   ; 
    }
    for (i32 k =0; k<num_active_l;++k){
        dual_for_eq += z_p(active_set_l(k)) * C_copy.row(active_set_l(k)) ;
        tmp2_l(k) =  residual_in_z_l(active_set_l(k)) - z_p(active_set_l(k))/ mu_in   ; 
    }
    for (i32 k =0; k<num_inactive;++k){
        tmp3(k) = z_p(inactive_set(k)) ; 
    }

    Scalar b0 = 2 * ( d_dual_for_eq.dot(dual_for_eq) + tmp_d2_u.dot(tmp2_u) + tmp_d2_l.dot(tmp2_l) +  tmp3.dot(tmp_d3) + primal_residual_eq.dot(d_primal_residual_eq) ) ; 
    //std::cout << "b0 " << b0 << std::endl;
    // c0 computation
    Scalar c0 = dual_for_eq.squaredNorm() + tmp2_u.squaredNorm() + tmp3.squaredNorm() + tmp2_l.squaredNorm() + primal_residual_eq.squaredNorm() ; 
    //std::cout << "c0 " << c0 << std::endl;
    // derivation of the loss function value and corresponding argmin alpha

    auto res = Scalar(0);
    
    if (a0!=0){
        alpha = (-b0/(2*a0)) ; 
        res = a0 * pow(alpha, Scalar(2)) + b0 * alpha + c0 ; 
    } else if (b0!= 0){
        alpha = (-c0/(b0)) ; 
        res = b0 * alpha + c0;
    }
    else{
        alpha = 0 ; 
        res = c0;
    }
    
    //// unit test : check the value is the same as the squared norm of the gradient
    // cannot be derived with other function as positive part with given alpha, is not the same as the one used for deriving the gradient norm
    
    /*
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> positive_part_z =  (positive_part).select(z_e + alpha * dz_, Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(n_in));

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp_unit_test(dim+n_eq+n_in); 
    
    tmp_unit_test.topRows(dim) = dual_for_eq + alpha * d_dual_for_eq;

    std::cout << "norm diff for dual part " << tmp_unit_test.topRows(dim).squaredNorm()  - d_dual_for_eq.squaredNorm() * pow(alpha, Scalar(2)) - 2*  d_dual_for_eq.dot(dual_for_eq) * alpha - dual_for_eq.squaredNorm()<< std::endl << std::endl ;

    tmp_unit_test.middleRows(dim, n_eq) = primal_residual_eq + alpha *d_primal_residual_eq ; 

    std::cout << "norm diff for equality part " << tmp_unit_test.middleRows(dim, n_eq).squaredNorm()  - d_primal_residual_eq.squaredNorm() * pow(alpha, Scalar(2)) - 2*  d_primal_residual_eq.dot(primal_residual_eq) * alpha - primal_residual_eq.squaredNorm()<< std::endl << std::endl ;

    for (i32 k =0; k<num_active;++k){
        tmp_unit_test(dim+n_eq+k) =  residual_in_z(active_set(k)) + (Cdx(active_set(k))*alpha  - (positive_part_z(active_set(k)) )/ mu_in) ; 
    }
    

    for (i32 k =0; k<num_inactive;++k){
        tmp_unit_test(dim+n_eq+num_active+k) = positive_part_z(inactive_set(k)) ; 
    }

    std::cout << "norm diff for active inequality part " << tmp_unit_test.middleRows(dim+n_eq,num_active).squaredNorm()  - tmp_d2.squaredNorm() * pow(alpha, Scalar(2)) - 2*  tmp_d2.dot(tmp2) * alpha - tmp2.squaredNorm()<< std::endl << std::endl ;
    std::cout << "norm diff for inactive inequality part " << tmp_unit_test.middleRows(dim+n_eq+num_active,num_inactive).squaredNorm()  - tmp_d3.squaredNorm() * pow(alpha, Scalar(2)) - 2*  tmp_d3.dot(tmp3) * alpha - tmp3.squaredNorm()<< std::endl << std::endl ;

    std::cout << "norm diff " << tmp_unit_test.squaredNorm()  - res << std::endl << std::endl ;
    */

	return res;
}

template <
		typename Scalar>
auto initial_guess_line_search( //
		VectorView<Scalar> x,
		VectorView<Scalar> y,
        VectorView<Scalar> ze,
        VectorView<Scalar> dw,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		qp::QpView<Scalar> qp) -> Scalar {
    
    Scalar machine_eps = std::numeric_limits<Scalar>::epsilon();
    Scalar machine_inf = std::numeric_limits<Scalar>::infinity() ;

	i32 dim = qp.H.rows;
	i32 n_eq = qp.A.rows;
    i32 n_in = qp.C.rows;

	auto H = (qp.H).to_eigen();
	auto A = (qp.A).to_eigen();
	auto b = (qp.b).to_eigen();
	auto C = (qp.C).to_eigen();
	auto d = (qp.d).to_eigen();

	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_e = ze.to_eigen();
    auto dx_ = dw.to_eigen().head(dim);
    auto dy_ = dw.to_eigen().middleRows(dim,n_eq);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dz_ = dw.to_eigen().tail(n_in);
    
    Scalar alpha = 1 ; 

    std::list<Scalar> alphas = {} ; // TODO mettre un vecteur 
    // add solutions of equation z+alpha dz = 0

    for (i32 i = 0 ; i < n_in ; i++){
        if ( std::abs(z_e(i)) != 0){
            alphas.push_back( -z_e(i)/(dz_(i)+machine_eps) ) ;
        }
    }

    // add solutions of equation C(x+alpha dx)-d +ze/mu_in = 0

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Cdx = C * dx_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual_in_z = C * x_ -d+z_e/mu_in ; 
    
    for (i32 i = 0 ; i < n_in ; i++){
        if (  std::abs(Cdx(i))  !=0 ){
            alphas.push_back(-residual_in_z(i)/(Cdx(i)+ machine_eps)) ; 
        }
    }
    
    // prepare all needed algebra for gradient norm computation (and derive them only once) --> to add later in a workspace in qp_solve function

    //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = qp::detail::to_eigen_vector(qp.g); 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = (qp.g).to_eigen();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_for_eq = H*x_ + g + A.transpose() * y_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_dual_for_eq = H*dx_ + A.transpose() * dy_  + rho * dx_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_primal_residual_eq = A * dx_ - dy_/mu_eq;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_residual_eq = A * x_ - b ;
    
    // define intervals with these solutions

    if (alphas.empty() == false){
        alphas.sort();
        alphas.unique();

        // for each node active set and associated gradient is computed

        std::list<Scalar> liste_norm_grad_noeud = {} ;

        for (auto a : alphas) {
           
            if ( std::abs(a) < 10000){ 
                
                // calcul de la norm du gradient du noeud 
                Scalar grad_norm = gradient_norm_computation(ze, dz_, mu_eq, mu_in, rho, qp.C, Cdx, residual_in_z, d_dual_for_eq, dual_for_eq, d_primal_residual_eq, primal_residual_eq, a,dim,n_eq,n_in);
                //std::cout << "a " << a << " grad " << grad_norm << std::endl << std::endl ;
                liste_norm_grad_noeud.push_back(grad_norm) ; 
            } 
            else{
                liste_norm_grad_noeud.push_back(machine_inf);
            }
        }

        // define intervals with alphas

        std::list<Scalar> liste_norm_grad_interval = {} ;
        std::list<Scalar> liste_argmin = {} ; 

        std::list<Scalar> interval =alphas ;
        alphas.push_front( (interval.front()-Scalar(1)) ) ; 
        alphas.push_back( (interval.back()+Scalar(1)) ) ; 

        std::vector<Scalar> intervals{ std::begin(interval), std::end(interval) };

        i32 n_ = intervals.size() ; 
        for(i32 i = 0; i < n_-1; ++i) {

            // on va calculer l'active set de l'intervalle en utilisant le point milieu de cet intervalle
            Scalar a = (intervals[i]+intervals[i+1])/2 ; 
            // on calcule le minimum associé à cet invervalle, le minimum alpha est changé inplace par a
            Scalar associated_grad_2_norm = local_saddle_point(ze, dz_, mu_eq, mu_in,rho, qp.C, Cdx, residual_in_z, d_dual_for_eq, dual_for_eq, d_primal_residual_eq, primal_residual_eq, a,dim,n_eq,n_in) ;
            // si le minimum est dans l'intervalle on le prend en compte
            if (i == 0){
                if (a <= intervals[1]){
                    liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                    liste_argmin.push_back(a) ;
                }
            }
            else if (i == n_-2){
                if (a >= intervals[n_-2]){
                        liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                        liste_argmin.push_back(a) ;
                }
            }
            else{
                if (a <= intervals[i+1] && intervals[i] <= a ){
                        liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                        liste_argmin.push_back(a) ;
                }
            }
        }

        if (liste_norm_grad_interval.empty() == false){

            std::vector<Scalar> vec_norm_grad_interval{ std::begin(liste_norm_grad_interval), std::end(liste_norm_grad_interval) };
            std::vector<Scalar> vec_argmin{ std::begin(liste_argmin), std::end(liste_argmin) };

            auto index = std::min_element(vec_norm_grad_interval.begin(),vec_norm_grad_interval.end()) - vec_norm_grad_interval.begin(); 
            
            alpha = vec_argmin[index] ;
            //std::cout << " min interval " << vec_norm_grad_interval[index] << std::endl << std::endl ;
        }
        else if (liste_norm_grad_noeud.empty() == false ){

            std::vector<Scalar> vec_alphas{ std::begin(alphas), std::end(alphas) };
            std::vector<Scalar> vec_norm_grad_noeud{ std::begin(liste_norm_grad_noeud), std::end(liste_norm_grad_noeud) };

            auto index = std::min_element(vec_norm_grad_noeud.begin(),vec_norm_grad_noeud.end()) - vec_norm_grad_noeud.begin();  
            alpha = vec_alphas[index] ; 
        }  
    }
    
   return alpha;
    
}

template <typename Scalar>
auto initial_guess_line_search_box( //
		VectorView<Scalar> x,
		VectorView<Scalar> y,
        VectorView<Scalar> ze,
        VectorView<Scalar> dw,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		qp::QpViewBox<Scalar> qp) -> Scalar {
    /* 
    Considerning the following qp = (H, g, A, b, C, u,l) and a Newton step (dx,dy,dz) the fonction gives one optimal alpha minizing the L2 norm of the following vector

    H.dot(x) + g + rho_primal_proximal * (x-xe) + A.transpose().dot(y) +  C[active_inequalities_l,:].transpose().dot(z[active_inequalities_l]) + C[active_inequalities_u,:].transpose().dot(z[active_inequalities_u])
    (residual_eq * mu_eq - (y-ye) )/mu_eq
    (residual_in_u[active_inequalities_u] * mu_in  - (z[active_inequalities_u]-ze[active_inequalities_u]))/mu_in
    (residual_in_l[active_inequalities_l] * mu_in - (z[active_inequalities_l]-ze[active_inequalities_l]))/mu_in
    z[inactive_inequalities]

    with 
        x = xe + alpha dx ; y = ye + alpha dy ; z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u],0) and z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l],0)
    
    Furthermore 
        residual_eq = A.dot(x) - b
        residual_in_u = C.dot(x) - u
        residual_in_l = C.dot(x) - l
        active_inequalities_u = residual_in_u + alpha Cdx >=0
        active_inequalities_l = residual_in_l + alpha Cdx <=0
        active_inequalities = active_inequalities_u + active_inequalities_l
        inactive_inequalities = ~active_inequalities  

    It can be shown that when one optimal active set is found for the qp problem, then the optimal alpha canceling (hence minimizing) the L2 norm of the merit function is unique and equal to 1
    If the optimal active set is not found, one optimal alpha found can not deviate new iterates formed from the sub problem solution

    To do so the algorithm has the following structure :

        1/ 
            1.1/ it computes the "nodes" alpha which cancel  C.dot(xe+alpha dx) - u,  C.dot(xe+alpha dx) - l and ze + alpha dz 
            1.2/ it prepares all needed algebra in order not to derive it each time (TODO : integrate it at a higher level in the solver)
        2/ 
            2.1/ it sorts the alpha nodes 
            2.2/ for each "node" it derives the L2 norm of the vector to minimize (see function: gradient_norm_computation_box) and stores it

        3/ it defines all intervals on which the active set is constant 
            3.1/ it  define intervals (for ex with n+1 nodes): [alpha[0]-1;alpha[0]],[alpha[0],alpha[1]], ....; [alpha[n],alpha[n]+1]]
            3.2/ for each interval 
                it derives the mean node (alpha[i]+alpha[i+1])/2 and the corresponding active sets active_inequalities_u and active_inequalities_l
                cap ze and dz
                    (indeed optimal lagrange multiplier z satisfy z[active_inequalities_u] = max((ze+alpha dz)[active_inequalities_u],0) and and z[active_inequalities_l] = min((ze+alpha dz)[active_inequalities_l],0)
            3.3/ on this interval the merit function is a second order polynomial in alpha
                the function "local_saddle_point_box" derives the exact minimum and corresponding merif function L2 norm (for this minimum)
                
            3.4/ if the argmin is within the interval [alpha[i],alpha[i+1]] is stores the argmin and corresponding L2 norm
        
        4/ if the list of argmin obtained from intervals is not empty the algorithm return the one minimizing the most the merit function
           Otherwise, it returns the node minimizing the most the merit function
    */
    
    Scalar machine_eps = std::numeric_limits<Scalar>::epsilon();
    Scalar machine_inf = std::numeric_limits<Scalar>::infinity() ;

	i32 dim = qp.H.rows;
	i32 n_eq = qp.A.rows;
    i32 n_in = qp.C.rows;

	auto H = (qp.H).to_eigen();
	auto A = (qp.A).to_eigen();
	auto b = (qp.b).to_eigen();
	auto C = (qp.C).to_eigen();
	auto u = (qp.u).to_eigen();
    auto l = (qp.l).to_eigen();

	auto x_ = x.to_eigen();
	auto y_ = y.to_eigen();
	auto z_e = ze.to_eigen();
    auto dx_ = dw.to_eigen().head(dim);
    auto dy_ = dw.to_eigen().middleRows(dim,n_eq);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dz_ = dw.to_eigen().tail(n_in);

    Scalar alpha = 1 ; 

    /////////// STEP 1 ////////////
    //computing the "nodes" alphas which cancel  C.dot(xe+alpha dx) - u,  C.dot(xe+alpha dx) - l and ze + alpha dz  /////////////
   
    std::list<Scalar> alphas = {} ; // TODO use a vector instead of a list
    // 1.1 add solutions of equation z+alpha dz = 0

    for (i32 i = 0 ; i < n_in ; i++){
        if ( std::abs(z_e(i)) != 0){
            alphas.push_back( -z_e(i)/(dz_(i)+machine_eps) ) ;
        }
    }

    // 1.1 add solutions of equations C(x+alpha dx)-u +ze/mu_in = 0 and C(x+alpha dx)-l +ze/mu_in = 0

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Cdx = C * dx_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual_in_z_u = C * x_ -u+z_e/mu_in ; 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual_in_z_l = C * x_ -l+z_e/mu_in ; 

    for (i32 i = 0 ; i < n_in ; i++){
        if (  std::abs(Cdx(i))  !=0 ){
            alphas.push_back(-residual_in_z_u(i)/(Cdx(i)+ machine_eps)) ; 
            alphas.push_back(-residual_in_z_l(i)/(Cdx(i)+ machine_eps)) ; 
        }
    }
    
    // 1.2 it prepares all needed algebra in order not to derive it each time (TODO : integrate it at a higher level in the solver)

    //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = qp::detail::to_eigen_vector(qp.g); 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = (qp.g).to_eigen(); 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dual_for_eq = H*x_ + g + A.transpose() * y_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_dual_for_eq = H*dx_ + A.transpose() * dy_  + rho * dx_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> d_primal_residual_eq = A * dx_ - dy_/mu_eq;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> primal_residual_eq = A * x_ - b ;
    
    if (alphas.empty() == false){
        //////// STEP 2 ////////
        // 2.1/ it sort alpha nodes
        alphas.sort();
        alphas.unique();

        //2.2/ for each node active set and associated gradient is computed

        std::list<Scalar> liste_norm_grad_noeud = {} ;

        for (auto a : alphas) {
           
            if ( std::abs(a) < 1.E6){ 
                
                // calcul de la norm du gradient du noeud 
                //std::cout << "a " << a <<  std::endl; 
                Scalar grad_norm = gradient_norm_computation_box(ze, dz_, mu_eq, mu_in, rho, qp.C, Cdx, residual_in_z_u,residual_in_z_l, d_dual_for_eq, dual_for_eq, d_primal_residual_eq, primal_residual_eq, a,dim,n_eq,n_in);
                std::cout << "c++ a " << a << "associated grad_norm " << grad_norm << std::endl; 
                liste_norm_grad_noeud.push_back(grad_norm) ; 
            } 
            else{
                liste_norm_grad_noeud.push_back(machine_inf);
            }
        }

        //////////STEP 3 ////////////
        // 3.1 : define intervals with alphas

        std::list<Scalar> liste_norm_grad_interval = {} ;
        std::list<Scalar> liste_argmin = {} ; 

        std::list<Scalar> interval = alphas ; 
        interval.push_front( (alphas.front()-Scalar(1)) ) ; 
        interval.push_back( (alphas.back()+Scalar(1)) ) ; 

        std::vector<Scalar> intervals{ std::begin(interval), std::end(interval) };
        i32 n_ = intervals.size() ; 
        /*
        for (i32 i = 0; i < n_; ++i){
            std::cout << "i " << i << " intervals(i) " << intervals[i] << std::endl;
        */
       
        for(i32 i = 0; i < n_-1; ++i) {
            
            // 3.2 : it derives the mean node (alpha[i]+alpha[i+1])/2 
            // the corresponding active sets active_inequalities_u and active_inequalities_l cap ze and dz is derived through function local_saddle_point_box

            Scalar a_ = (intervals[i]+intervals[i+1])/Scalar(2.0); 

            //3.3 on this interval the merit function is a second order polynomial in alpha
            //   the function "local_saddle_point_box" derives the exact minimum and corresponding merif function L2 norm (for this minimum)
            Scalar associated_grad_2_norm = local_saddle_point_box(ze, dz_, mu_eq, mu_in,rho, qp.C, Cdx, residual_in_z_u, residual_in_z_l, d_dual_for_eq, dual_for_eq, d_primal_residual_eq, primal_residual_eq, a_,dim,n_eq,n_in) ;
            
            //std::cout << "c++ (intervals[i]+intervals[i+1])/2.0 " << (intervals[i]+intervals[i+1])/Scalar(2.0) << " a_ after " << a_ << " associated_grad_2_norm : " << associated_grad_2_norm << std::endl;

            //3.4 if the argmin is within the interval [alpha[i],alpha[i+1]] is stores the argmin and corresponding L2 norm

            if (i == 0){
                if ( a_ <= intervals[1]){
                    liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                    liste_argmin.push_back(a_) ;
                }
            }
            else if (i == n_-2){
                if (a_ >= intervals[n_-2]){
                        liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                        liste_argmin.push_back(a_) ;
                }
            }
            else{
                if ( a_ <= intervals[i+1] && intervals[i] <= a_){
                        liste_norm_grad_interval.push_back(associated_grad_2_norm) ;
                        liste_argmin.push_back(a_) ;
                }
            }
        }

        ///////// STEP 4 ///////////
        //if the list of argmin obtained from intervals is not empty the algorithm return the one minimizing the most the merit function
        // Otherwise, it returns the node minimizing the most the merit function

        if (liste_norm_grad_interval.empty() == false){

            std::vector<Scalar> vec_norm_grad_interval{ std::begin(liste_norm_grad_interval), std::end(liste_norm_grad_interval) };
            std::vector<Scalar> vec_argmin{ std::begin(liste_argmin), std::end(liste_argmin) };
            /*
            i32 n__ = vec_norm_grad_interval.size() ; 
            for (i32 i = 0 ; i<n__ ;++i){
                        std::cout << "c++ vec_norm_grad_interval : " << vec_norm_grad_interval[i] << " vec_argmin " << vec_argmin[i] << std::endl;
            }
            */
            auto index = std::min_element(vec_norm_grad_interval.begin(),vec_norm_grad_interval.end()) - vec_norm_grad_interval.begin(); 
            
            alpha = vec_argmin[index] ;
            //std::cout << " min interval " << vec_norm_grad_interval[index] << std::endl << std::endl ;
        }
        else if (liste_norm_grad_noeud.empty() == false ){

            std::vector<Scalar> vec_alphas{ std::begin(alphas), std::end(alphas) };
            std::vector<Scalar> vec_norm_grad_noeud{ std::begin(liste_norm_grad_noeud), std::end(liste_norm_grad_noeud) };

            /*
            i32 n___ = vec_alphas.size() ; 
            for (i32 i = 0 ; i<n___ ;++i){
                        std::cout << "c++ vec_alphas : " << vec_alphas[i] << " vec_norm_grad_noeud " << vec_norm_grad_noeud[i] << std::endl;
            }
            */

            auto index = std::min_element(vec_norm_grad_noeud.begin(),vec_norm_grad_noeud.end()) - vec_norm_grad_noeud.begin();  
            alpha = vec_alphas[index] ; 
        }  
    }
    
   return alpha;
    
}

template <
		typename Scalar>
auto correction_guess_line_search( //
		VectorView<Scalar> x,
        VectorView<Scalar> xe,
		VectorView<Scalar> ye,
        VectorView<Scalar> ze,
        VectorView<Scalar> dx,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		qp::QpView<Scalar> qp) -> Scalar {

    Scalar machine_eps = std::numeric_limits<Scalar>::epsilon();
    Scalar machine_inf = std::numeric_limits<Scalar>::infinity() ;

    i32 n_in = qp.C.rows;

    auto H = (qp.H).to_eigen();
    auto A = (qp.A).to_eigen();
	auto C = (qp.C).to_eigen();
	auto d = (qp.d).to_eigen();
    auto b = (qp.b).to_eigen();

	auto x_ = x.to_eigen();
	auto z_e = ze.to_eigen();
    auto y_e = ye.to_eigen();
    auto dx_ = dx.to_eigen();

    Scalar alpha = 1 ; 

    std::list<Scalar> alphas = {} ; 

    // add solutions of equation C(x+alpha dx)-d +ze/mu_in = 0

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  Cdx =  C * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  residual_in_z = (C * x_ -d+z_e/mu_in) ; 
    
    for (i32 i = 0 ; i < n_in ; i++){
        if (  std::abs(Cdx(i))  !=0 && residual_in_z(i) != 0){
            alphas.push_back(-residual_in_z(i)/(Cdx(i)+ machine_eps)) ; 
        }
    }

    // prepare all needed algebra for gradient norm computation (and derive them only once) --> to add later in a workspace in qp_solve function

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Hdx = H * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Adx = A * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual_in_y = A * x_ - b + y_e/mu_eq ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = (qp.g).to_eigen(); 

    // define intervals with these solutions

    if (alphas.empty() == false){
        alphas.sort();
        alphas.unique();

        Scalar last_neg_grad = 0 ;
        Scalar alpha_last_neg = 0 ;
        Scalar first_pos_grad = 0 ;
        Scalar alpha_first_pos = 0 ;

        for (auto a : alphas) {
            
            if ( a > 0){ 
                //std::cout << "a : " <<  a << std::endl << std::endl;
                if ( a  < 1.e7 ) {
                    
                    //line_search_result<Scalar> res = gradient_norm_qpalm(x, xe, dx, mu_eq, mu_in, rho, a, Hdx, g, Adx, residual_in_y, residual_in_z, Cdx,n_in) ;
                    Scalar gr = gradient_norm_qpalm(x, xe, dx, mu_eq, mu_in, rho, a, Hdx, g, Adx, residual_in_y, residual_in_z, Cdx,n_in) ;
                    /*
                    std::cout << "grad :" << gr << std::endl << std::endl ;
                    */
                    if(gr < 0){
                        alpha_last_neg = a ;
                        last_neg_grad = gr ; 
                    }else{
                        first_pos_grad = gr ;
                        alpha_first_pos = a ;
                        break ;
                    }
                    
                }
            }

        }
        //std::cout <<"last_neg_grad " << last_neg_grad << std::endl << std::endl;
        if (last_neg_grad == Scalar(0)){
            alpha_last_neg = Scalar(0);
            Scalar gr = gradient_norm_qpalm(x, xe, dx, mu_eq, mu_in, rho, Scalar(0), Hdx, g, Adx, residual_in_y, residual_in_z, Cdx,n_in) ;
            last_neg_grad = gr ; 
        }

        alpha = alpha_last_neg - last_neg_grad * (alpha_first_pos -alpha_last_neg ) / (first_pos_grad-last_neg_grad);

    }
    //std::cout <<"biding alpha " << alpha << std::endl << std::endl;
    return alpha ; 
}


template <
		typename Scalar>
auto correction_guess_line_search_box( //
		VectorView<Scalar> x,
        VectorView<Scalar> xe,
		VectorView<Scalar> ye,
        VectorView<Scalar> ze,
        VectorView<Scalar> dx,
        Scalar mu_eq,
        Scalar mu_in,
        Scalar rho,
		qp::QpViewBox<Scalar> qp) -> Scalar {


    /*
    The function follows the algorithm designed by qpalm (see algorithm 2 : https://arxiv.org/pdf/1911.02934.pdf)

    To do so it does the following steps
    1/ 
        1.1/ Store solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha dx)-u +ze/mu_in = 0
        1.2/ Prepare all needed algebra for gradient computation (and derive them only once) (TODO to add at a higher level in the solver)
        1.3/ Sort the alphas
    2/ 
        2.1/
            For each positive alpha compute the first derivative of phi(alpha) = [proximal augmented lagrangian of the subproblem evaluated at x_k + alpha dx] using function "gradient_norm_qpalm_box"
                (By construction for alpha = 0,  phi'(alpha) <= 0 and phi'(alpha) goes to infinity with alpha hence it cancels uniquely at one optimal alpha*)
            
            while phi'(alpha)<=0 store the derivative (noted last_grad_neg) and alpha (last_alpha_neg)
            the first time phi'(alpha) > 0 store the derivative (noted first_grad_pos) and alpha (first_alpha_pos), and break the loop

        2.2/
            If first_alpha_pos corresponds to the first positive alpha of previous loop 
                then do 
                    last_alpha_neg = 0 and last_grad_neg = phi'(0) using function "gradient_norm_qpalm_box" 
        
        2.3/
            the optimal alpha is within the interval [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi' is an affine function in alpha
                alpha* = alpha_last_neg - last_neg_grad * (alpha_first_pos -alpha_last_neg ) / (first_pos_grad-last_neg_grad);

    */

    Scalar machine_eps = std::numeric_limits<Scalar>::epsilon();
    Scalar machine_inf = std::numeric_limits<Scalar>::infinity() ;

    i32 n_in = qp.C.rows;

    auto H = (qp.H).to_eigen();
    auto A = (qp.A).to_eigen();
	auto C = (qp.C).to_eigen();
	auto u = (qp.u).to_eigen();
	auto l = (qp.l).to_eigen();
    auto b = (qp.b).to_eigen();

	auto x_ = x.to_eigen();
	auto z_e = ze.to_eigen();
    auto y_e = ye.to_eigen();
    auto dx_ = dx.to_eigen();

    Scalar alpha = 1 ; 

    std::list<Scalar> alphas = {} ; 

    ///////// STEP 1 /////////
    // 1.1 add solutions of equations C(x+alpha dx)-l +ze/mu_in = 0 and C(x+alpha dx)-u +ze/mu_in = 0

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  Cdx =  C * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  residual_in_z_u = (C * x_ -u+z_e/mu_in) ; 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>  residual_in_z_l = (C * x_ -l+z_e/mu_in) ; 

    for (i32 i = 0 ; i < n_in ; i++){
        if (  Cdx(i)  !=0 ){
            alphas.push_back(-residual_in_z_u(i)/(Cdx(i)+ machine_eps)) ; 
        }
        if (  Cdx(i)  !=0 ){
            alphas.push_back(-residual_in_z_l(i)/(Cdx(i)+ machine_eps)) ; 
        }
    }

    // 1.2 prepare all needed algebra for gradient norm computation (and derive them only once) --> to add later in a workspace in qp_solve function

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Hdx = H * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Adx = A * dx_ ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual_in_y = A * x_ - b + y_e/mu_eq ;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> g = (qp.g).to_eigen(); 

    

    if (alphas.empty() == false){
        // 1.3 sort the alphas
        alphas.sort();
        alphas.unique();

        ////////// STEP 2 ///////////

        Scalar last_neg_grad = 0 ;
        Scalar alpha_last_neg = 0 ;
        Scalar first_pos_grad = 0 ;
        Scalar alpha_first_pos = 0 ;

        for (auto a : alphas) {
            
            if ( a > 0){ 
                //std::cout << "a : " <<  a << std::endl << std::endl;
                if ( a  < 1.e7 ) {
                    
                    /*
                    2.1/
                    For each positive alpha compute the first derivative of phi(alpha) = [proximal augmented lagrangian of the subproblem evaluated at x_k + alpha dx] using function "gradient_norm_qpalm_box"
                        (By construction for alpha = 0,  phi'(alpha) <= 0 and phi'(alpha) goes to infinity with alpha hence it cancels uniquely at one optimal alpha*)
                    
                    while phi'(alpha)<=0 store the derivative (noted last_grad_neg) and alpha (last_alpha_neg)
                    the first time phi'(alpha) > 0 store the derivative (noted first_grad_pos) and alpha (first_alpha_pos), and break the loop
                    */
                    Scalar gr = gradient_norm_qpalm_box(x, xe, dx, mu_eq, mu_in, rho, a, Hdx, g, Adx, residual_in_y, residual_in_z_u,residual_in_z_l, Cdx,n_in) ;
                    
                    /*
                    std::cout << "a :" << a  << " grad :" << gr << std::endl ;
                    std::cout << "grad : " <<  gr << std::endl << std::endl;
                    */
                    if(gr < 0){
                        alpha_last_neg = a ;
                        last_neg_grad = gr ; 
                    }else{
                        first_pos_grad = gr ;
                        alpha_first_pos = a ;
                        break ;
                    }
                    
                }
            }

        }
        //std::cout <<"last_neg_grad " << last_neg_grad << std::endl << std::endl;

        /*
        2.2/
            If first_alpha_pos corresponds to the first positive alpha of previous loop 
            then do 
                last_alpha_neg = 0 and last_grad_neg = phi'(0) using function "gradient_norm_qpalm_box" 
        */
        if (last_neg_grad == Scalar(0)){
            alpha_last_neg = Scalar(0);
            Scalar gr = gradient_norm_qpalm_box(x, xe, dx, mu_eq, mu_in, rho, alpha_last_neg, Hdx, g, Adx, residual_in_y, residual_in_z_u,residual_in_z_l, Cdx,n_in) ;
            last_neg_grad = gr ; 
        }

        /*
        2.3/
            the optimal alpha is within the interval [last_alpha_neg,first_alpha_pos] and can be computed exactly as phi' is an affine function in alpha
        */
        alpha = alpha_last_neg - last_neg_grad * (alpha_first_pos -alpha_last_neg ) / (first_pos_grad-last_neg_grad);

    }
    //std::cout <<"biding alpha " << alpha << std::endl << std::endl;
    return alpha ; 
}

auto activeSetChange( //
        Eigen::Matrix<bool, Eigen::Dynamic, 1>& new_active_set,
        //Eigen::Ref<Eigen::Matrix<bool, Eigen::Dynamic, 1> const> new_active_set,
		//Eigen::Matrix<i32, Eigen::Dynamic, 1>& current_bijection_map, ldlt::view does not work with type bool, only Scalar...
        Eigen::Matrix<i32, Eigen::Dynamic, 1>& current_bijection_map,
        i32 n_c,
		i32 n,
        i32 n_eq,
        i32 n_in
        ) -> Eigen::Matrix<i32, Eigen::Dynamic, 1> {

    /*
    arguments :
        1/ new_active_set : a vector which contains new active set of the problem, namely if
            new_active_set_u = Cx_k-u +z_k/mu_in>= 0
            new_active_set_l = Cx_k-l +z_k/mu_in<= 0
            
            then new_active_set = new_active_set_u OR new_active_set_l
        2/ current_bijection_map : a vector for which each entry corresponds to the current row of C of the current factorization

            for example, naming C_initial the initial C matrix of the problem, and C_current the one of the current factorization, then

            C_initial[i,:] = C_current[current_bijection_mal[i],:] for all i
        
        3/ n_c : the current number of active_inequalities

    This algorithm ensures that for all new version of C_current in the LDLT factorization all row index i < n_c correspond to current active indexes (all other correspond to inactive rows)

    To do so, 
        1/ for initialization 
            1.1/ new_bijection_map = current_bijection_map 
            1.2/ n_c_f = n_c
        
        2/ All active indexes of the current bijection map (i.e current_bijection_map(i) < n_c by assumption) which are not active anymore in the new active set (new_active_set(i)=false)
            are put at the end of new_bijection_map, i.e
                
                2.1/ for all j if new_bijection_map(j) > new_bijection_map(i), then new_bijection_map(j)-=1;
                2.2/ n_c_f -=1
                2.3/ new_bijection_map(i) = n_in-1 ;
        3/ All active indexe of the new active set (new_active_set(i) == true) which are not active in the new_bijection_map (new_bijection_map(i) >= n_c_f) 
            are put at the end of the current version of C, i.e

                3.1/ if new_bijection_map(j) < new_bijection_map(i) && new_bijection_map(j) >= n_c_f then new_bijection_map(j)+=1
                3.2/ new_bijection_map(i) = n_c_f 
                3.3/ n_c_f +=1 

    It returns finally the new_bijection_map, for which new_bijection_map(n_in) = n_c_f
    */
    i32 n_c_f = n_c ;
    Eigen::Matrix<i32, Eigen::Dynamic, 1> new_bijection_map(n_in+1)  ;
    //new_bijection_map.array().topRows(n_in) = current_bijection_map.cast<i32>();
    new_bijection_map.array().topRows(n_in) = current_bijection_map;


    // suppression pour le nouvel active set, ajout dans le nouvel unactive set

    for (i32 i = 0 ; i < n_in ; i++){
        if (current_bijection_map(i) < n_c){
            //std::cout << " new_active_set(i) " << new_active_set(i) << std::endl <<std::endl;
            if (new_active_set(i) == false){
                //i32 pos = new_bijection_map(i) ; 
                //S_f,D_f = remove_col_and_row(S_f,D_f,int(pos)+n+n_eq); to add
                for (i32 j = 0 ; j < n_in ; j++) { 
                    if (new_bijection_map(j) > new_bijection_map(i)){
                        new_bijection_map(j)-=1;
                    }
                }
                n_c_f -=1 ;
                new_bijection_map(i) = n_in-1 ;
                //std::cout << " new_bijection_map " << new_bijection_map << "n_c_f " << n_c_f <<std::endl <<std::endl;
            }
        }
    }

    // ajout au nouvel active set, suppression pour le nouvel unactive set

    for (i32 i = 0 ; i < n_in ; i++){
        if (new_active_set(i) == true){
            if (new_bijection_map(i) >= n_c_f){ 

                //J_i = np.zeros(n_c_f+1) ;
                //J_i(n_c_f) = -1./mu_in ;
                //row = np.concatenate([qp_scaled_init[4].toarray()[i,:],np.zeros(n_eq),J_i],axis=None) ;
                //S_f,D_f = add_col_and_row(S_f,D_f,row) ; // To add
                for (i32 j = 0 ; j < n_in ; j++){
                    if (new_bijection_map(j) < new_bijection_map(i) && new_bijection_map(j) >= n_c_f){
                        new_bijection_map(j)+=1 ; 
                    }
                }
                new_bijection_map(i) = n_c_f ; 
                //std::cout << " new_bijection_map " << new_bijection_map << std::endl <<std::endl;
                n_c_f +=1 ; 
            }
        }
    }

    //std::cout << "current_bijection_map before copy " << current_bijection_map << std::endl << std::endl;
    //current_bijection_map = new_bijection_map ; 
    //std::cout << "current_bijection_map after copy " << current_bijection_map << " n_c_f " << n_c_f << std::endl << std::endl;
    // return --> in place change ? S_f, D_f // l_active_set_f : in place change ; return n_c_f

    new_bijection_map(n_in) = n_c_f;
    return new_bijection_map;
}

} // namespace line_search

} // namespace qp

#endif /* end of include guard INRIA_LDLT_LINE_SEARCH_HPP_HDWGZKCLS */
