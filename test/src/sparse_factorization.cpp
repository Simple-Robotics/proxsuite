//
// Copyright (c) 2022 INRIA
//
#include <proxsuite/linalg/sparse/factorize.hpp>
#include <proxsuite/linalg/sparse/update.hpp>
#include <proxsuite/linalg/sparse/rowmod.hpp>
#include <proxsuite/linalg/veg/vec.hpp>
#include <doctest.hpp>
#include <iostream>

template<typename T, typename I>
auto
to_eigen(proxsuite::linalg::sparse::MatRef<T, I> a) noexcept
  -> Eigen::Matrix<T, -1, -1>
{
  return a.to_eigen();
}

template<typename T, typename I>
auto
to_eigen_vec(proxsuite::linalg::sparse::VecRef<T, I> v) noexcept
  -> Eigen::Matrix<T, -1, 1>
{
  Eigen::Matrix<T, -1, 1> out(v.nrows());
  out.setZero();
  for (proxsuite::linalg::veg::isize p = 0; p < v.nnz(); ++p) {
    out.data()[proxsuite::linalg::sparse::util::zero_extend(
      v.row_indices()[p])] = v.values()[p];
    // proxsuite::linalg::sparse::util::zero_extend : converti en type usize
    //  usize :unsigned
    //  isize :négatif ou positive
    //  v.row_indices() : pointeur vers liste des indices non nul
    //  out.data(): pointeur premier élt de la sortie
  }
  //
  return out;
}
// slice: view vers un vecteur
// perm[0] : 1er elet du vecteur
template<typename I>
auto
to_eigen_perm(proxsuite::linalg::veg::Slice<I> perm)
  -> Eigen::PermutationMatrix<-1, -1, I>
{
  Eigen::PermutationMatrix<-1, -1, I> perm_eigen;
  perm_eigen.indices().resize(perm.len());
  std::memmove( //
    perm_eigen.indices().data(),
    perm.ptr(),
    proxsuite::linalg::veg::usize(perm.len()) * sizeof(I));
  // copie perm.ptr() vers perm_eigen ...
  //  proxsuite::linalg::veg::usize(perm.len()) * sizeof(I) :taille de la zone à
  //  copier
  return perm_eigen;
}

template<typename T, typename I>
auto
reconstruct_with_perm(proxsuite::linalg::veg::Slice<I> perm_inv,
                      proxsuite::linalg::sparse::MatRef<T, I> ld)
  -> Eigen::Matrix<T, -1, -1, Eigen::ColMajor>
{
  using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
  Mat ld_eigen = to_eigen(ld);
  auto perm_inv_eigen = to_eigen_perm(perm_inv);
  Mat l = ld_eigen.template triangularView<Eigen::UnitLower>();
  Mat d = ld_eigen.diagonal().asDiagonal();
  Mat ldlt = l * d * l.transpose();
  return perm_inv_eigen.inverse() * ldlt * perm_inv_eigen;
}

template<typename T, typename I>
auto
ldlt_with_perm(proxsuite::linalg::veg::Slice<I> perm_inv,
               proxsuite::linalg::sparse::MatRef<T, I> a)
  -> Eigen::Matrix<T, -1, -1, Eigen::ColMajor>
{
  using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;

  VEG_ASSERT(a.nrows() == a.ncols());

  proxsuite::linalg::veg::isize n = a.nrows();

  Eigen::PermutationMatrix<-1, -1, I> perm_inv_eigen = to_eigen_perm(perm_inv);

  auto ld_perm_eigen =
    Mat((perm_inv_eigen *
         Mat(::to_eigen(a).template selfadjointView<Eigen::Upper>()) *
         perm_inv_eigen.inverse())
          .template triangularView<Eigen::Lower>());
  {
    for (proxsuite::linalg::veg::isize i = 0; i < n; ++i) {
      auto a12 = ld_perm_eigen.row(i).head(i).transpose();
      auto l11 = ld_perm_eigen.topLeftCorner(i, i)
                   .template triangularView<Eigen::UnitLower>();
      auto l12 = ld_perm_eigen.row(i).head(i).transpose();
      auto d1 = ld_perm_eigen.diagonal().head(i);
      l12 = l11.solve(a12);
      l12 = d1.asDiagonal().inverse() * l12;
      ld_perm_eigen(i, i) -= l12.dot(d1.asDiagonal() * l12);
    }
  }
  return ld_perm_eigen;
}

using namespace proxsuite::linalg::sparse;
using namespace proxsuite::linalg::veg;

TEST_CASE("ldlt: factorize compressed")
{
  using I = int;
  using T = long double;

  isize n = 11;
  isize nnz = 27;

  Vec<I> col_ptrs;
  Vec<I> row_ind;
  Vec<T> vals;

  /*
  https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
  Values: vals	22	7	3	5	14	1	17	8
  InnerIndices:row_ind	1	2	0	2	4	2	1
  4 OuterStarts:col_ptrs	0	2	4	5	6	8
  */

  // csc format
  for (auto c : { 0, 1, 2, 4, 5, 6, 9, 11, 14, 16, 21, 27 }) {
    col_ptrs.push(I(c)); // l'indice du premier  élt de chaque colonne non null
                         // de chaque de colonne
  }

  for (auto r : { 0, 1, 1, 2, 3, 4, 0, 3, 5, 0, 6, 1, 4, 7,
                  5, 8, 2, 3, 5, 7, 9, 2, 4, 6, 7, 9, 10 }) {
    row_ind.push(I(r));
  }
  for (isize i = 0; i < nnz; ++i) {
    vals.push(1);
  }
  for (isize i = 0; i < n; ++i) {
    // sort in decreasing order so eigen doesn't permute them
    vals[col_ptrs[i + 1] - 1] = T(20) - T(i);
  }

  auto a = MatRef<T, I>{
    from_raw_parts,
    n,
    n,
    nnz,
    col_ptrs.ptr(),
    nullptr, // si compressé --> nullptr; si non compressé, il faut un ptr vers
             // nbr d'elt non nul par colonne
    row_ind.ptr(),
    vals.ptr(),
  };

  {
    auto req = etree_req(
      Tag<I>{},
      n); // calcule la mémoire pour la fonction etree (arbre d'élimination)
    req = req | postorder_req(
                  proxsuite::linalg::veg::Tag<I>{},
                  n); // postorder_req: calcule mémoire pour fonction postorder
                      // (calcule certaine permutation de la matrice a)
    req = req | column_counts_req(Tag<I>{},
                                  n,
                                  nnz); // column_counts_req :column_counts  nbr
                                        // d'elt non nul par colonne de la ldlt
    // | : any of operator
    // & : all of operator
    Vec<unsigned char> stack_data; // vector de bytes
    stack_data.resize_for_overwrite(
      req.alloc_req()); // resize pour avoir taille du workspace necessaire
    auto stack = dynstack::DynStackMut{
      from_slice_mut, stack_data.as_mut()
    }; // stack: view sur ce vecteur

    auto const parent_expected = [] {
      Vec<I> parent;
      for (auto p : { 5, 2, 7, 5, 7, 6, 8, 9, 9, 10, -1 }) {
        parent.push(I(p));
      }
      return parent;
    }();

    Vec<I> parent; // parent de chaque elt dans l'arbre d'elimination
    parent.resize_for_overwrite(n);

    etree(parent.ptr_mut(),
          a.symbolic(),
          stack); // calcule l'arbre d'élimination; l'arbre est définit par une
                  // liste de parent
    // check : https://youtu.be/uZKJPTo4dZs check course on sparse factorization

    auto post_expected = [] {
      Vec<I> post;
      for (auto p : { 1, 2, 4, 7, 0, 3, 5, 6, 8, 9, 10 }) {
        post.push(I(p));
      }
      return post;
    }(); // post order expected

    Vec<I> post;
    post.resize_for_overwrite(n);
    postorder(post.ptr_mut(), parent.ptr(), n, stack); // calculer post order
    // postorder: equivalente à la permutation de l'identité du point de vue du
    // ratio de sparsité ; garde meme structure matrice, mais change l'ordre
    // pour que les parents de chaque noeud vienne après leur enfant
    auto const counts_expected = [&] {
      Vec<I> counts;
      for (auto c : { 3, 3, 4, 3, 3, 4, 4, 3, 3, 2, 1 }) {
        counts.push(I(c));
      }
      return counts;
    }(); // nbr d'elts non nul par colonne de la ldlt

    Vec<I> counts;
    counts.resize_for_overwrite(n);
    column_counts( //
      counts.ptr_mut(),
      a.symbolic(),
      parent.ptr(),
      post.ptr(),
      stack); // calcle nbr d'elts non nul par colonne de la ldlt
    for (isize i = 0; i < n; ++i) {
      CHECK(counts[i] == counts_expected[i]);
    }
  }

  {
    // VERSio minimal format compressée
    Vec<I> l_col_ptrs;
    Vec<I> etree;
    l_col_ptrs.resize_for_overwrite(
      n + 1); // n+1 car: pour chaque colonne tu as besoin de l'indice du début
              // de la colonne et fin de la colonne
    etree.resize_for_overwrite(n);
    // factorize_symbolic_req : calcule mémoire de la factorization symbolique
    // Ordering:: ammd ou no permutation ou user_provided
    Vec<unsigned char> _stack;
    _stack.resize_for_overwrite(
      (factorize_symbolic_req(
         proxsuite::linalg::veg::Tag<I>{}, n, nnz, Ordering::amd) |
       factorize_numeric_req(proxsuite::linalg::veg::Tag<T>{},
                             proxsuite::linalg::veg::Tag<I>{},
                             n,
                             nnz,
                             Ordering::amd))
        .alloc_req());
    dynstack::DynStackMut stack{ from_slice_mut, _stack.as_mut() };

    Vec<I> perm_inv;
    perm_inv.resize_for_overwrite(n);

    factorize_symbolic_col_counts( // calcule la taille de chaque colonne de la
                                   // factorization (matrice maximale a; donne
                                   // taille maximale de chaque colonne de l)
      l_col_ptrs.ptr_mut(),        // indices des colonnes
      etree.ptr_mut(),
      perm_inv.ptr_mut(),
      static_cast<I*>(
        nullptr), // permutation direct: si elle null on calcule la permutation
                  // inverse avec amd ; si non null on utilise la permutation
                  // directe ppur calculer la permutation inverse
      a.symbolic(),
      stack);

    auto lnnz = isize(util::zero_extend(
      l_col_ptrs[n])); // l_col_ptrs[n] : nbr d'elts non nul total

    Vec<I> l_row_indices;
    Vec<T> l_values;
    l_row_indices.resize_for_overwrite(lnnz);
    l_values.resize_for_overwrite(lnnz);

    factorize_numeric(l_values.ptr_mut(),
                      l_row_indices.ptr_mut(),
                      nullptr,
                      nullptr,
                      l_col_ptrs.ptr(),
                      etree.ptr(),
                      perm_inv.ptr(),
                      a,
                      stack);
    //
    MatRef<T, I> l{
      from_raw_parts,
      n,
      n,
      nnz,
      l_col_ptrs.ptr(),
      {}, // nbr d'elets non nul par colonne ; la matrice est en format
          // compressé
      l_row_indices.ptr(),
      l_values.ptr(),
    };

    auto ld_eigen = to_eigen(l);
    CHECK((ld_eigen - ldlt_with_perm(perm_inv.as_ref(), a)).norm() < T(1e-10));
  }
}

TEST_CASE("ldlt: factorize uncompressed, rank update")
{
  using I = isize;
  using T = double;

  isize n = 11;
  isize nnz = 27;
  // assez d"espace  dans la matrice non compressé:
  //
  Vec<I> col_ptrs_compressed;
  Vec<I> row_ind_compressed;
  Vec<T> vals_compressed;

  for (auto c : { 0, 1, 2, 4, 5, 6, 9, 11, 14, 16, 21, 27 }) {
    col_ptrs_compressed.push(I(c));
  }

  for (auto r : { 0, 1, 1, 2, 3, 4, 0, 3, 5, 0, 6, 1, 4, 7,
                  5, 8, 2, 3, 5, 7, 9, 2, 4, 6, 7, 9, 10 }) {
    row_ind_compressed.push(I(r));
  }
  for (isize i = 0; i < nnz; ++i) {
    vals_compressed.push(T(i));
  }
  for (isize i = 0; i < n; ++i) {
    // sort in decreasing order so eigen doesn't permute them
    vals_compressed[col_ptrs_compressed[i + 1] - 1] = 10 * (T(20) - T(i));
  }

  Vec<I> col_ptrs; // format non compressé
  Vec<I> nnz_per_col;
  Vec<I> row_ind;
  Vec<T> vals;

  /*
  https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
  Values:  vals	22	7	_	3	5	14	_	_
  1 _	17	8 -->
  InnerIndices: row_ind	1	2	_	0	2	4	_
  _ 2	_	1	4 -->
  OuterStarts:col_ptrs	0	3	5	8	10	12 -->
  InnerNNZs: nnz_per_col
  */

  for (isize j = 0; j < n; ++j) {
    nnz_per_col.push(col_ptrs_compressed[j + 1] - col_ptrs_compressed[j]);
  }
  for (isize k = 0; k < n + 1; ++k) {
    col_ptrs.push(k * n); // indice de la ke colonne non nul
  }
  row_ind.resize_for_overwrite(n * n);
  vals.resize_for_overwrite(n * n);

  isize src_index = 0; // src -->source
  // copie matrice indice lignes et valeur du format compressé au non compressé
  for (isize j = 0; j < n; ++j) {
    isize dest_index = col_ptrs[j];
    for (isize k = 0; k < nnz_per_col[j]; ++k) {
      row_ind[dest_index] = row_ind_compressed[src_index];
      vals[dest_index] = vals_compressed[src_index];
      ++dest_index;
      ++src_index;
    }
  }

  auto a = MatRef<T, I>{
    from_raw_parts,    n,          n, nnz, col_ptrs.ptr(),
    nnz_per_col.ptr(), // version non compressée
    row_ind.ptr(),     vals.ptr(),
  };
  // version minimal pour factorization au format non compressé
  Vec<I> l_nnz_per_col;
  Vec<I> l_col_ptrs;
  Vec<I> l_row_indices;
  Vec<T> l_values;

  Vec<I> etree;
  Vec<I> perm_inv;

  l_nnz_per_col.resize_for_overwrite(n);
  for (isize k = 0; k < n + 1; ++k) {
    l_col_ptrs.push(k * n);
  }
  l_row_indices.resize_for_overwrite(n * n);
  l_values.resize_for_overwrite(n * n);

  etree.resize_for_overwrite(n);
  perm_inv.resize_for_overwrite(n);

  Vec<unsigned char> _stack;
  _stack.resize_for_overwrite(
    (factorize_symbolic_req(
       proxsuite::linalg::veg::Tag<I>{}, n, nnz, Ordering::amd) |
     factorize_numeric_req(proxsuite::linalg::veg::Tag<T>{},
                           proxsuite::linalg::veg::Tag<I>{},
                           n,
                           nnz,
                           Ordering::amd))
      .alloc_req());
  dynstack::DynStackMut stack{ from_slice_mut, _stack.as_mut() };

  factorize_symbolic_non_zeros(l_nnz_per_col.ptr_mut(),
                               etree.ptr_mut(),
                               perm_inv.ptr_mut(),
                               {},
                               a.symbolic(),
                               stack);
  factorize_numeric(
    l_values.ptr_mut(),
    l_row_indices.ptr_mut(),
    nullptr,
    nullptr, // si on veut ajouter une diag à la matrice triangulaire supérieur
             // on peut le faire avec diag_to_add et  la permuation direct perm;
             // permutation direct
    l_col_ptrs.ptr(),
    etree.ptr(),
    perm_inv.ptr(),
    a,
    stack);
  //
  isize lnnz = 0;
  for (isize k = 0; k < n; ++k) {
    lnnz += l_nnz_per_col[k];
  }

  // ld :matrice triangulaire inférieur avec les n valeurs de l et la diagonale
  // remplacée par d
  MatMut<T, I> ld{
    from_raw_parts,
    n,
    n,
    lnnz,
    l_col_ptrs.ptr_mut(),
    l_nnz_per_col.ptr_mut(),
    l_row_indices.ptr_mut(),
    l_values.ptr_mut(),
  };

  auto ld_eigen = to_eigen(ld.as_const());
  std::cout << "ld_eigen " << ld_eigen << std::endl;
  CHECK((ld_eigen - ldlt_with_perm(perm_inv.as_ref(), a)).norm() < T(1e-10));

  Vec<T> w_values; // vecteur à ajouter pour rank update
  Vec<I> w_row_indices;
  for (auto i : { 5, 6, 8 }) {
    w_row_indices.push(i);
  }
  for (auto v : { 1., 2., 3. }) {
    w_values.push(v);
  }

  VecRef<T, I> w{
    from_raw_parts, n, w_values.len(), w_row_indices.ptr(), w_values.ptr(),
  };
  using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
  auto w_eigen = to_eigen_vec(w);
  ld = rank1_update(ld,
                    etree.ptr_mut(),
                    perm_inv.ptr(),
                    w,
                    1.0,
                    stack); // alpha * w w.T et alpha = 1.0 ici
  CHECK((reconstruct_with_perm(perm_inv.as_ref(), ld.as_const()) -
         Mat(to_eigen(a).selfadjointView<Eigen::Upper>()) -
         w_eigen * w_eigen.transpose())
          .norm() < T(1e-10));
}

TEST_CASE("ldlt: row mod")
{
  using I = isize;
  using T = double;

  isize n = 11;
  isize nnz = 27;

  Vec<I> col_ptrs_compressed;
  Vec<I> row_ind_compressed;
  Vec<T> vals_compressed;

  for (auto c : { 0, 1, 2, 4, 5, 6, 9, 11, 14, 16, 21, 27 }) {
    col_ptrs_compressed.push(I(c));
  }

  for (auto r : { 0, 1, 1, 2, 3, 4, 0, 3, 5, 0, 6, 1, 4, 7,
                  5, 8, 2, 3, 5, 7, 9, 2, 4, 6, 7, 9, 10 }) {
    row_ind_compressed.push(I(r));
  }
  for (isize i = 0; i < nnz; ++i) {
    vals_compressed.push(T(i));
  }
  for (isize i = 0; i < n; ++i) {
    // sort in decreasing order so eigen doesn't permute them
    vals_compressed[col_ptrs_compressed[i + 1] - 1] = 10 * (T(20) - T(i));
  }

  Vec<I> col_ptrs;
  Vec<I> nnz_per_col;
  Vec<I> row_ind;
  Vec<T> vals;
  for (isize j = 0; j < n; ++j) {
    nnz_per_col.push(col_ptrs_compressed[j + 1] - col_ptrs_compressed[j]);
  }
  for (isize k = 0; k < n + 1; ++k) {
    col_ptrs.push(k * n);
  }
  row_ind.resize_for_overwrite(n * n);
  vals.resize_for_overwrite(n * n);

  isize src_index = 0;
  for (isize j = 0; j < n; ++j) {
    isize dest_index = col_ptrs[j];
    for (isize k = 0; k < nnz_per_col[j]; ++k) {
      row_ind[dest_index] = row_ind_compressed[src_index];
      vals[dest_index] = vals_compressed[src_index];
      ++dest_index;
      ++src_index;
    }
  }

  auto a = MatRef<T, I>{
    from_raw_parts, n,          n, nnz, col_ptrs.ptr(), nnz_per_col.ptr(),
    row_ind.ptr(),  vals.ptr(),
  };

  Vec<I> l_nnz_per_col;
  Vec<I> l_col_ptrs;
  Vec<I> l_row_indices;
  Vec<T> l_values;

  Vec<I> etree;
  Vec<I> perm_inv;

  l_nnz_per_col.resize_for_overwrite(n);
  for (isize k = 0; k < n + 1; ++k) {
    l_col_ptrs.push(k * n);
  }
  l_row_indices.resize_for_overwrite(n * n);
  l_values.resize_for_overwrite(n * n);

  etree.resize_for_overwrite(n);
  perm_inv.resize_for_overwrite(n);

  Vec<unsigned char> _stack;
  _stack.resize_for_overwrite(
    (factorize_symbolic_req(
       proxsuite::linalg::veg::Tag<I>{}, n, nnz, Ordering::amd) |
     factorize_numeric_req(proxsuite::linalg::veg::Tag<T>{},
                           proxsuite::linalg::veg::Tag<I>{},
                           n,
                           nnz,
                           Ordering::amd))
      .alloc_req());
  dynstack::DynStackMut stack{ from_slice_mut, _stack.as_mut() };

  factorize_symbolic_non_zeros(l_nnz_per_col.ptr_mut(),
                               etree.ptr_mut(),
                               perm_inv.ptr_mut(),
                               {},
                               a.symbolic(),
                               stack);
  factorize_numeric(l_values.ptr_mut(),
                    l_row_indices.ptr_mut(),
                    nullptr,
                    nullptr,
                    l_col_ptrs.ptr(),
                    etree.ptr(),
                    perm_inv.ptr(),
                    a,
                    stack);

  isize lnnz = 0;
  for (isize k = 0; k < n; ++k) {
    lnnz += l_nnz_per_col[k];
  }

  MatMut<T, I> ld{
    from_raw_parts,
    n,
    n,
    lnnz,
    l_col_ptrs.ptr_mut(),
    l_nnz_per_col.ptr_mut(),
    l_row_indices.ptr_mut(),
    l_values.ptr_mut(),
  };

  auto ld_eigen = to_eigen(ld.as_const());

  CHECK((ld_eigen - ldlt_with_perm(perm_inv.as_ref(), a)).norm() < T(1e-10));

  auto dump_reconstructed = [&] {
    std::cout << (reconstruct_with_perm(perm_inv.as_ref(), ld.as_const()))
              << '\n'
              << '\n';
  };
  std::cout << to_eigen(ld.as_const()) << '\n' << '\n';

  dump_reconstructed();
  ld = delete_row(ld, etree.ptr_mut(), perm_inv.ptr(), 2, stack);
  dump_reconstructed();
  ld = delete_row(ld, etree.ptr_mut(), perm_inv.ptr(), 6, stack);
  dump_reconstructed();

  Vec<T> w_values;
  Vec<I> w_row_indices;
  for (auto i : { 1, 9, 10 }) {
    w_row_indices.push(i);
  }
  for (auto v : { 1300., 1200., -1. }) {
    w_values.push(v);
  }

  VecRef<T, I> w{
    from_raw_parts, n, w_values.len(), w_row_indices.ptr(), w_values.ptr(),
  };

  ld =
    add_row(ld,
            etree.ptr_mut(),
            perm_inv.ptr(),
            2,
            w,
            180,
            stack); //  2: indice de la novuelle colonne et 180: l'elt diagonale
  std::cout << to_eigen(ld.as_const()) << '\n' << '\n';
  dump_reconstructed();
}
