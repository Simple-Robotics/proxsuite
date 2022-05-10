#ifndef SPARSE_LDLT_FACTORIZE_HPP_P6FLZUBLS
#define SPARSE_LDLT_FACTORIZE_HPP_P6FLZUBLS

#include "linearsolver/sparse/core.hpp"
#include <Eigen/OrderingMethods>

namespace linearsolver {
namespace sparse {

// y += a x
template <typename T, typename I>
void axpy(DenseVecMut<T> y, MatRef<T, I> a, DenseVecRef<T> x) noexcept(false) {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			y.nrows() == a.nrows(),
			x.nrows() == a.ncols());

	auto py = y.as_slice_mut().ptr_mut();
	auto px = x.as_slice().ptr();
	I const* pai = a.row_indices().ptr();
	T const* pax = a.values().ptr();

	for (usize j = 0; j < usize(a.ncols()); ++j) {

		auto col_start = a.col_start(j);
		auto col_end = a.col_end(j);

		for (usize p = col_start; p < col_end; ++p) {
			auto i = util::zero_extend(pai[p]);
			py[i] += pax[p] * x[j];
		}
	}
}

// y += a.T x
template <typename T, typename I>
void atxpy(DenseVecMut<T> y, MatRef<T, I> a, DenseVecRef<T> x) noexcept(false) {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			a.is_compressed(),
			y.nrows() == a.ncols(),
			x.nrows() == a.nrows());

	auto py = y.as_slice_mut().ptr_mut();
	auto px = x.as_slice().ptr();
	I const* pai = a.row_indices().ptr();
	T const* pax = a.values().ptr();

	for (usize j = 0; j < usize(a.ncols()); ++j) {
		auto col_start = a.col_start(j);
		auto col_end = a.col_end(j);

		T sum = 0;
		for (usize p = col_start; p < col_end; ++p) {
			auto i = util::zero_extend(pai[p]);
			sum += pax[p] * px[i];
		}
		py[j] += sum;
	}
}

template <typename I>
auto transpose_req(veg::Tag<I> /*tag*/, isize nrows) noexcept
		-> veg::dynstack::StackReq {
	return {nrows * isize(sizeof(I)), isize(alignof(I))};
}

// at = a.T
template <typename T, typename I>
void transpose( //
		MatMut<T, I> at,
		MatRef<T, I> a,
		DynStackMut stack) noexcept(VEG_CONCEPT(nothrow_copyable<T>)) {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			at.is_compressed(),
			at.nrows() == a.ncols(),
			at.ncols() == a.nrows(),
			at.nnz() == a.nnz());

	auto pai = a.row_indices().ptr();
	auto pax = a.values().ptr();

	auto patp = at.col_ptrs_mut().ptr_mut();
	auto pati = at.row_indices_mut().ptr_mut();
	auto patx = at.values_mut().ptr_mut();

	auto _work = stack.make_new(veg::Tag<I>{}, at.ncols()).unwrap();
	auto work = _work.as_mut().ptr_mut();

	// work[i] = num zeros in ith row of A
	if (a.nnz_per_col().ptr() == nullptr) {
		for (usize p = 0; p < usize(a.nnz()); ++p) {
			util::wrapping_inc(mut(work[util::zero_extend(pai[p])]));
		}
	} else {
		for (usize j = 0; j < a.ncols(); ++j) {
			isize col_start = a.col_start(j);
			isize col_end = a.col_end(j);
			for (isize p = col_start; p < col_end; ++p) {
				util::wrapping_inc(mut(work[util::zero_extend(pai[p])]));
			}
		}
	}

	// compute the cumulative sum
	for (usize j = 0; j < usize(at.ncols()); ++j) {
		patp[j + 1] = util::checked_non_negative_plus(patp[j], work[j]);
		work[j] = patp[j];
	}

	for (usize j = 0; j < usize(a.ncols()); ++j) {
		auto col_start = a.col_start(j);
		auto col_end = a.col_end(j);
		for (usize p = col_start; p < col_end; ++p) {
			auto i = util::zero_extend(pai[p]);
			auto q = util::zero_extend(work[i]);

			pati[q] = j;
			patx[q] = pax[p];
			util::wrapping_inc(mut(work[i]));
		}
	}
}

template <typename I>
auto transpose_symbolic_req(veg::Tag<I> /*tag*/, isize nrows) noexcept
		-> veg::dynstack::StackReq {
	return {nrows * isize(sizeof(I)), isize(alignof(I))};
}

template <typename I>
void transpose_symbolic( //
		SymbolicMatMut<I> at,
		SymbolicMatRef<I> a,
		DynStackMut stack) noexcept {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			at.is_compressed(),
			at.nrows() == a.ncols(),
			at.ncols() == a.nrows(),
			at.nnz() == a.nnz());

	auto pai = a.row_indices().ptr();

	auto patp = at.col_ptrs_mut().ptr_mut();
	auto pati = at.row_indices_mut().ptr_mut();

	auto _work = stack.make_new(veg::Tag<I>{}, at.ncols()).unwrap();
	auto work = _work.as_mut().ptr_mut();

	// work[i] = num zeros in ith row of A
	for (usize p = 0; p < usize(a.nnz()); ++p) {
		util::wrapping_inc(mut(work[util::zero_extend(pai[p])]));
	}

	// compute the cumulative sum
	for (usize j = 0; j < usize(at.ncols()); ++j) {
		patp[j + 1] = util::checked_non_negative_plus(patp[j], work[j]);
		work[j] = patp[j];
	}

	for (usize j = 0; j < usize(a.ncols()); ++j) {
		auto col_start = a.col_start(j);
		auto col_end = a.col_end(j);
		for (usize p = col_start; p < col_end; ++p) {
			auto i = util::zero_extend(pai[p]);
			auto q = util::zero_extend(work[i]);

			pati[q] = I(j);
			util::wrapping_inc(mut(work[i]));
		}
	}
}

// l is unit lower triangular
// solve x <- l \ x
template <typename T, typename I>
void dense_lsolve(DenseVecMut<T> x, MatRef<T, I> l) noexcept(false) {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			l.nrows() == l.ncols(),
			x.nrows() == l.nrows()
			/* l is unit lower triangular */
	);

	usize n = usize(l.nrows());

	auto pli = l.row_indices().ptr();
	auto plx = l.values().ptr();

	auto px = x.as_slice_mut().ptr_mut();

	for (usize j = 0; j < n; ++j) {
		auto const xj = px[j];
		auto col_start = l.col_start(j);
		auto col_end = l.col_end(j);

		// skip the diagonal entry
		for (usize p = col_start + 1; p < col_end; ++p) {
			auto i = util::zero_extend(pli[p]);
			px[i] -= plx[p] * xj;
		}
	}
}

// l is unit lower triangular
// solve x <- l.T \ x
template <typename T, typename I>
void dense_ltsolve(DenseVecMut<T> x, MatRef<T, I> l) noexcept(false) {
	using namespace _detail;

	VEG_ASSERT_ALL_OF( //
			l.nrows() == l.ncols(),
			x.nrows() == l.nrows()
			/* l is unit lower triangular */
	);

	usize n = usize(l.nrows());

	auto pli = l.row_indices().ptr();
	auto plx = l.values().ptr();

	auto px = x.as_slice_mut().ptr_mut();

	usize j = n;
	while (true) {
		if (j == 0) {
			break;
		}
		--j;

		auto col_start = l.col_start(j);
		auto col_end = l.col_end(j);
		T acc0 = 0;
		T acc1 = 0;
		T acc2 = 0;
		T acc3 = 0;

		// skip the diagonal entry
		usize pstart = col_start + 1;
		usize pcount = col_end - pstart;

		usize p = pstart;
		for (; p < pstart + pcount / 4 * 4; p += 4) {
			auto i0 = util::zero_extend(pli[p + 0]);
			auto i1 = util::zero_extend(pli[p + 1]);
			auto i2 = util::zero_extend(pli[p + 2]);
			auto i3 = util::zero_extend(pli[p + 3]);
			acc0 += plx[p + 0] * px[i0];
			acc1 += plx[p + 1] * px[i1];
			acc2 += plx[p + 2] * px[i2];
			acc3 += plx[p + 3] * px[i3];
		}
		for (; p < pstart + pcount; ++p) {
			auto i0 = util::zero_extend(pli[p + 0]);
			acc0 += plx[p + 0] * px[i0];
		}

		acc0 = (acc0 + acc1) + (acc2 + acc3);

		px[j] -= acc0;
	}
}

template <typename I>
auto etree_req(veg::Tag<I> /*tag*/, isize n) noexcept
		-> veg::dynstack::StackReq {
	return {n * isize{sizeof(I)}, alignof(I)};
}

// compute the elimination tree of the cholesky factor  of a
// a is considered symmetric but should only contain terms from the upper
// triangular part
template <typename I>
VEG_INLINE void etree( //
		SliceMut<I> parent,
		SymbolicMatRef<I> a,
		DynStackMut stack) noexcept {
	using namespace _detail;

	VEG_ASSERT(parent.len() == a.ncols());

	usize n = usize(a.ncols());
	auto pai = a.row_indices().ptr();

	auto pparent = parent.ptr_mut();
	auto _work = stack.make_new_for_overwrite(veg::Tag<I>{}, isize(n)).unwrap();
	auto pancestors = _work.as_mut().ptr_mut();

	// for each column of a
	for (usize k = 0; k < n; ++k) {
		pparent[k] = I(-1);
		pancestors[k] = I(-1);
		// assuming elimination subtree T_{k-1} is known, compute T_k

		auto col_start = a.col_start(k);
		auto col_end = a.col_end(k);

		// for each non zero element of a
		for (usize p = col_start; p < col_end; ++p) {
			// get the row
			auto i = util::zero_extend(pai[p]);
			// skip if looking at lower triangular half
			if (i >= k) {
				continue;
			}

			// go up towards the root of the tree
			usize node = i;
			auto next = usize(-1);
			while (true) {
				if (node == usize(-1) || node >= k) {
					break;
				}

				// use the highest known ancestor instead of the parent
				next = util::sign_extend(pancestors[node]);

				// set the highest known ancestor to k, since we know we're going
				// to find it eventually
				pancestors[node] = I(k);

				// if there is no highest known ancestor, we must have hit the root of
				// the tree
				// set the parent to k
				if (next == usize(-1)) {
					pparent[node] = I(k);
					break;
				}
				// go to the highest ancestor
				node = next;
			}
		}
	}
}

namespace _detail {
inline auto ereach_req(isize k) noexcept -> veg::dynstack::StackReq {
	return {(k + 1) * isize{sizeof(bool)}, alignof(bool)};
}

// compute the set of reachable nodes from the non zero pattern of a_{.,k}
// not including the node k itself
template <typename I>
VEG_NODISCARD VEG_INLINE auto ereach(
		SliceMut<I> s,
		SymbolicMatRef<I> a,
		Slice<I> parent,
		isize k,
		bool* pmarked) noexcept -> SliceMut<I> {

	usize n = usize(a.ncols());
	auto k_ = usize(k);

	VEG_ASSERT_ALL_OF(
			n > 0,
			a.nrows() == n,
			k >= 0,
			k < n,
			parent.len() == n,
			// s big enough to hold all the nodes other than k
			s.len() + 1 >= k);

	auto pai = a.row_indices().ptr();
	auto pparent = parent.ptr();

	auto col_start = a.col_start(k_);
	auto col_end = a.col_end(k_);

	pmarked[k_] = true;

	usize top = n;

	// for each non zero element of a_{.,k}
	for (usize p = col_start; p < col_end; ++p) {
		auto i = util::zero_extend(pai[p]);

		// only triu part of a
		if (i > k_) {
			continue;
		}

		usize len = 0;
		while (true) {
			// if we reach a marked node
			if (pmarked[i]) {
				break;
			}

			// can't overwrite top of the stack since elements of s are unique
			// and s is large enough to hold all the nodes
			s[isize(len)] = I(i);
			util::wrapping_inc(mut(len));

			// mark node i as reached
			pmarked[i] = true;
			i = util::sign_extend(pparent[i]);
		}

		// make sure that we can memmove
		std::memmove( //
				s.ptr_mut() + (top - len),
				s.ptr(),
				usize(len) * sizeof(I));

		// move down the top of the stack
		top = util::wrapping_plus(top, -len);
	}

	for (usize q = top; q < n; ++q) {
		pmarked[s.ptr()[q]] = false;
	}
	pmarked[k_] = false;

	// [top, end[
	return s.split_at_mut(isize(top))[1_c];
}
} // namespace _detail

namespace _detail {
// return the next start_index
template <typename I>
VEG_INLINE auto postorder_depth_first_search( //
		SliceMut<I> post,
		usize root,
		usize start_index,
		I* pstack,
		I* pfirst_child,
		I* pnext_child) noexcept -> usize {
	using namespace _detail;

	auto ppost = post.ptr_mut();

	usize top = 0;
	pstack[0] = I(root);

	// stack is non empty
	while (top != usize(-1)) {
		auto current_node = util::zero_extend(pstack[top]);
		auto current_child = util::sign_extend(pfirst_child[current_node]);

		// no more children
		if (current_child == usize(-1)) {
			ppost[start_index] = I(current_node);
			++start_index;

			// pop node from the stack
			util::wrapping_dec(mut(top));
		} else {
			// add current child to the stack
			util::wrapping_inc(mut(top));
			pstack[top] = I(current_child);

			// next child is now the first child
			pfirst_child[current_node] = pnext_child[current_child];
		}
	}
	return start_index;
}
} // namespace _detail

template <typename I>
auto postorder_req(veg::Tag<I> /*tag*/, isize n) noexcept
		-> veg::dynstack::StackReq {
	return {(3 * n) * isize(sizeof(I)), alignof(I)};
}

template <typename I>
void postorder(SliceMut<I> post, Slice<I> parent, DynStackMut stack) noexcept {
	using namespace _detail;

	VEG_ASSERT(parent.len() == post.len());

	usize n = usize(parent.len());

	auto _work =
			stack.make_new_for_overwrite(veg::Tag<I>{}, 3 * isize(n)).unwrap();
	I* pwork = _work.as_mut().ptr_mut();

	I* pstack = pwork;
	I* pfirst_child = pstack + n;
	I* pnext_child = pfirst_child + n;

	// no children are found yet
	for (usize j = 0; j < n; ++j) {
		pfirst_child[j] = I(-1);
	}

	for (usize _j = 0; _j < n; ++_j) {
		// traverse in reverse order, since the children appear in reverse order
		// of insertion in the linked list
		usize j = n - 1 - _j;

		// if not a root node
		if (parent[isize(j)] != I(-1)) {
			// next child of this node is the previous first child
			pnext_child[j] = pfirst_child[util::zero_extend(parent[isize(j)])];
			// set this node to be the new first child
			pfirst_child[util::zero_extend(parent[isize(j)])] = I(j);
		}
	}

	usize start_index = 0;
	for (usize root = 0; root < n; ++root) {
		if (parent[isize(root)] == I(-1)) {
			start_index = _detail::postorder_depth_first_search(
					post, root, start_index, pstack, pfirst_child, pnext_child);
		}
	}
}

namespace _detail {
// returns -2 if j is not a leaf
// returns -1 if j is a first leaf
// returns the least common ancestor of j and the previous j otherwise
template <typename I>
VEG_INLINE auto least_common_ancestor(
		usize i,
		usize j,
		I const* pfirst,
		I* pmax_first,
		I* pprev_leaf,
		I* pancestor) noexcept -> I {
	using namespace _detail;

	// if upper triangular part, or not a leaf
	// leaves always have a new larger value of pfirst
	// add 1 to get the correct result when comparing with -1
	if (i <= j || util::wrapping_plus(pfirst[j], I(1)) <=
	                  util::wrapping_plus(pmax_first[i], I(1))) {
		return I(-2);
	}

	// update the largest max_first
	// no need to compare because the value of j increases
	// inbetween successive calls, and so does first_j
	pmax_first[i] = pfirst[j];

	// get the previous j
	usize j_prev = util::sign_extend(pprev_leaf[i]);

	// set the previous j to the current j
	pprev_leaf[i] = I(j);

	// if first leaf
	if (j_prev == usize(-1)) {
		return I(-1);
	}

	// else, subsequent leaf
	// get the least common ancestor of j and j_prev
	usize lca = j_prev;
	while (true) {
		if (lca == util::zero_extend(pancestor[lca])) {
			break;
		}
		lca = util::zero_extend(pancestor[lca]);
	}

	// compress the path to speed up the subsequent calls
	// to this function
	usize node = j_prev;
	while (true) {
		if (node == lca) {
			break;
		}
		usize next = util::zero_extend(pancestor[node]);
		pancestor[node] = I(lca);
		node = next;
	}

	return I(lca);
}
} // namespace _detail

template <typename I>
auto column_counts_req(veg::Tag<I> tag, isize n, isize nnz) noexcept
		-> veg::dynstack::StackReq {
	using veg::dynstack::StackReq;
	return StackReq{
						 isize{sizeof(I)} * (1 + 5 * n + nnz),
						 alignof(I),
				 } &
	       sparse::transpose_symbolic_req(tag, n);
}

template <typename I>
void column_counts(
		SliceMut<I> counts,
		SymbolicMatRef<I> a,
		Slice<I> parent,
		Slice<I> post,
		DynStackMut stack) noexcept {
	// https://youtu.be/uZKJPTo4dZs
	using namespace _detail;
	usize n = usize(a.nrows());
	VEG_ASSERT_ALL_OF( //
			a.ncols() == n,
			counts.len() == n);
	auto _at_work =
			stack.make_new_for_overwrite(veg::Tag<I>{}, 1 + 5 * isize(n) + a.nnz())
					.unwrap();
	auto pat_work = _at_work.ptr_mut();
	pat_work[0] = 0;
	pat_work[n] = I(a.nnz());

	SymbolicMatMut<I> at{
			from_raw_parts,
			isize(n),
			isize(n),
			a.nnz(),
			{unsafe, from_raw_parts, pat_work, isize(n + 1)},
			{unsafe, from_raw_parts, pat_work + n + 1, a.nnz()},
			{},
	};
	sparse::transpose_symbolic(at, a, stack);

	auto patp = at.col_ptrs().ptr();
	auto pati = at.row_indices().ptr();

	auto pwork = pat_work + n + 1 + a.nnz();

	auto pdelta = counts.ptr_mut();

	auto pfirst = pwork;
	auto pmax_first = pwork + n;
	auto pprev_leaf = pwork + 2 * n;
	auto pancestor = pwork + 3 * n;

	auto pcounts = counts.ptr_mut();
	auto ppost = post.ptr();
	auto pparent = parent.ptr();

	for (usize i = 0; i < 3 * n; ++i) {
		pwork[i] = I(-1);
	}
	for (usize i = 0; i < n; ++i) {
		pancestor[i] = I(i);
	}

	// for each column in a
	for (usize k = 0; k < n; ++k) {
		// in postordered fashion
		auto j = util::zero_extend(ppost[k]);

		// if first_j isn't computed, j must be a leaf
		// because if it's not a leaf then first_j will be initialized from
		// the init loop of its first descendant
		//
		// in which case initialize delta_j to 1
		pdelta[j] = (pfirst[j] == I(-1)) ? I(1) : I(0);

		// init loop
		while (true) {
			// while j is not a root, and the first descendant of j isn't computed
			// set the first descendant of j to k, as well as all of its ancestors
			// that don't yet have a first descendant
			if (j == usize(-1) || pfirst[j] != I(-1)) {
				break;
			}
			pfirst[j] = I(k);
			j = util::sign_extend(pparent[j]);
		}
	}

	// for each node
	for (usize k = 0; k < n; ++k) {
		// in postordered fashion
		auto j = util::zero_extend(ppost[k]);

		// if this node is the child of some other node
		if (pparent[j] != I(-1)) {
			// decrement the delta of that node
			// corresponding to the correction term e_j
			util::wrapping_dec(mut(pdelta[util::zero_extend(pparent[j])]));
		}

		auto col_start = util::zero_extend(patp[j]);
		auto col_end = util::zero_extend(patp[j + 1]);

		// iterate over lower triangular half of a
		for (usize p = col_start; p < col_end; ++p) {
			auto i = util::zero_extend(pati[p]);
			I lca = _detail::least_common_ancestor( //
					i,
					j,
					pfirst,
					pmax_first,
					pprev_leaf,
					pancestor);

			// if j is a leaf of T^i
			if (lca != I(-2)) {
				util::wrapping_inc(mut(pdelta[j]));

				// if j is a subsequent leaf
				if (lca != I(-1)) {
					util::wrapping_dec(mut(pdelta[util::zero_extend(lca)]));
				}
			}
		}

		if (pparent[j] != -1) {
			// set the ancestor of j
			pancestor[j] = pparent[j];
		}
	}

	// sum up the deltas
	for (usize j = 0; j < n; ++j) {
		if (parent[isize(j)] != I(-1)) {
			pcounts[util::zero_extend(parent[isize(j)])] = util::wrapping_plus(
					pcounts[util::zero_extend(parent[isize(j)])], pcounts[j]);
		}
	}
}

template <typename I>
auto amd_req(veg::Tag<I> /*tag*/, isize /*n*/, isize nnz) noexcept
		-> veg::dynstack::StackReq {
	return {nnz * isize(sizeof(char)), alignof(char)};
}

template <typename I>
void amd(SliceMut<I> perm, SymbolicMatRef<I> mat, DynStackMut stack) noexcept {
	// TODO: reimplement amd under BSD-3
	// https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/master/AMD

	isize n = perm.len();
	isize nnz = mat.nnz();

	VEG_ASSERT_ALL_OF( //
			mat.nrows() == n,
			mat.ncols() == n);

	Eigen::PermutationMatrix<-1, -1, I> perm_eigen;
	auto _ = stack.make_new(veg::Tag<char>{}, nnz).unwrap();

	Eigen::AMDOrdering<I>{}(
			Eigen::Map<Eigen::SparseMatrix<char, Eigen::ColMajor, I> const>{
					n,
					n,
					nnz,
					mat.col_ptrs().ptr(),
					mat.row_indices().ptr(),
					_.ptr(),
					mat.nnz_per_col().ptr(),
			}
					.template selfadjointView<Eigen::Upper>(),

			perm_eigen);
	std::memmove(
			perm.ptr_mut(), perm_eigen.indices().data(), usize(n) * sizeof(I));
}

namespace _detail {
template <typename I>
void inv_perm(SliceMut<I> perm_inv, Slice<I> perm) noexcept {
	auto n = usize(perm.len());
	auto pperm = perm.ptr();
	auto pperm_inv = perm_inv.ptr_mut();
	for (usize i = 0; i < n; ++i) {
		pperm_inv[util::zero_extend(pperm[i])] = I(i);
	}
}

template <typename I>
auto symmetric_permute_symbolic_req(veg::Tag<I> /*tag*/, isize n) noexcept
		-> veg::dynstack::StackReq {
	return {n * isize{sizeof(I)}, alignof(I)};
}
template <typename I>
auto symmetric_permute_req(veg::Tag<I> /*tag*/, isize n) noexcept
		-> veg::dynstack::StackReq {
	return {n * isize{sizeof(I)}, alignof(I)};
}

template <typename I>
void symmetric_permute_common(
		usize n,
		I const* pperm_inv,
		SymbolicMatRef<I> old_a,
		I* pnew_ap,
		I* pcol_counts) {
	for (usize old_j = 0; old_j < n; ++old_j) {
		usize new_j = util::zero_extend(pperm_inv[old_j]);

		auto col_start = old_a.col_start(old_j);
		auto col_end = old_a.col_end(old_j);

		for (usize p = col_start; p < col_end; ++p) {
			usize old_i = util::zero_extend(old_a.row_indices().ptr()[p]);

			if (old_i <= old_j) {
				usize new_i = util::zero_extend(pperm_inv[old_i]);
				util::wrapping_inc(mut(pcol_counts[new_i > new_j ? new_i : new_j]));
			}
		}
	}

	pnew_ap[0] = I(0);
	for (usize i = 0; i < n; ++i) {
		pnew_ap[i + 1] =
				util::checked_non_negative_plus(pnew_ap[i], pcol_counts[i]);
		pcol_counts[i] = pnew_ap[i];
	}
}

template <typename I>
void symmetric_permute_symbolic(
		SymbolicMatMut<I> new_a,
		SymbolicMatRef<I> old_a,
		Slice<I> perm_inv,
		DynStackMut stack) noexcept {

	usize n = usize(perm_inv.len());
	auto _work = stack.make_new(veg::Tag<I>{}, isize(n)).unwrap();
	I* pcol_counts = _work.as_mut().ptr_mut();

	VEG_ASSERT(new_a.is_compressed());
	auto pold_ap = old_a.col_ptrs().ptr();
	auto pold_ai = old_a.row_indices().ptr();

	auto pnew_ap = new_a.col_ptrs_mut().ptr_mut();
	auto pnew_ai = new_a.row_indices_mut().ptr_mut();

	auto pperm_inv = perm_inv.ptr();

	_detail::symmetric_permute_common(n, pperm_inv, old_a, pnew_ap, pcol_counts);

	auto pcurrent_row_index = pcol_counts;

	for (usize old_j = 0; old_j < n; ++old_j) {
		usize new_j = util::zero_extend(pperm_inv[old_j]);

		auto col_start = util::zero_extend(pold_ap[old_j]);
		auto col_end = util::zero_extend(pold_ap[old_j + 1]);

		for (usize p = col_start; p < col_end; ++p) {
			usize old_i = util::zero_extend(pold_ai[p]);

			if (old_i <= old_j) {
				usize new_i = util::zero_extend(pperm_inv[old_i]);

				usize new_max = new_i > new_j ? new_i : new_j;
				usize new_min = new_i < new_j ? new_i : new_j;

				auto row_idx = pcurrent_row_index[new_max];
				pnew_ai[row_idx] = I(new_min);
				pcurrent_row_index[new_max] = util::wrapping_plus(row_idx, I(1));
			}
		}
	}
}

template <typename T, typename I>
void symmetric_permute(
		MatMut<T, I> new_a,
		MatRef<T, I> old_a,
		Slice<I> perm_inv,
		DynStackMut stack) noexcept(VEG_CONCEPT(nothrow_copyable<T>)) {
	usize n = usize(perm_inv.len());
	auto _work = stack.make_new(veg::Tag<I>{}, isize(n)).unwrap();
	I* pcol_counts = _work.as_mut().ptr_mut();

	VEG_ASSERT(new_a.is_compressed());
	auto pold_ap = old_a.col_ptrs().ptr();
	auto pold_ai = old_a.row_indices().ptr();

	auto pnew_ap = new_a.col_ptrs_mut().ptr_mut();
	auto pnew_ai = new_a.row_indices_mut().ptr_mut();

	auto pperm_inv = perm_inv.ptr();

	_detail::symmetric_permute_common(
			n, pperm_inv, old_a.symbolic(), pnew_ap, pcol_counts);

	auto pcurrent_row_index = pcol_counts;

	auto pold_ax = old_a.values().ptr();
	auto pnew_ax = new_a.values_mut().ptr_mut();
	for (usize old_j = 0; old_j < n; ++old_j) {
		usize new_j = util::zero_extend(pperm_inv[old_j]);

		auto col_start = util::zero_extend(pold_ap[old_j]);
		auto col_end = util::zero_extend(pold_ap[old_j + 1]);

		for (usize p = col_start; p < col_end; ++p) {
			usize old_i = util::zero_extend(pold_ai[p]);

			if (old_i <= old_j) {
				usize new_i = util::zero_extend(pperm_inv[old_i]);

				usize new_max = new_i > new_j ? new_i : new_j;
				usize new_min = new_i < new_j ? new_i : new_j;

				auto row_idx = pcurrent_row_index[new_max];
				pnew_ai[row_idx] = I(new_min);
				pnew_ax[row_idx] = pold_ax[p];
				pcurrent_row_index[new_max] = util::wrapping_plus(row_idx, I(1));
			}
		}
	}
}
} // namespace _detail

enum struct Ordering : unsigned char {
	natural,
	user_provided,
	amd,
	ENUM_END,
};

template <typename I>
auto factorize_symbolic_req(
		veg::Tag<I> tag, isize n, isize nnz, Ordering o) noexcept
		-> veg::dynstack::StackReq {
	using veg::dynstack::StackReq;
	constexpr isize sz{sizeof(I)};
	constexpr isize al{alignof(I)};

	StackReq perm_req{0, al};
	StackReq amd_req{0, al};
	switch (o) {
	case Ordering::natural:
		break;
	case Ordering::amd:
		amd_req =
				StackReq{n * sz, al} & StackReq{sparse::amd_req(tag, n, nnz)};
		HEDLEY_FALL_THROUGH;
	case Ordering::user_provided:
		perm_req = perm_req & StackReq{(n + 1 + nnz) * sz, al};
		perm_req = perm_req & _detail::symmetric_permute_symbolic_req(tag, n);
	default:
		break;
	}

	StackReq parent_req = {n * sz, al};
	StackReq post_req = {n * sz, al};

	StackReq etree_req = sparse::etree_req(tag, n);
	StackReq postorder_req = sparse::postorder_req(tag, n);
	StackReq colcount_req = sparse::column_counts_req(tag, n, nnz);

	return amd_req              //
	       | (perm_req          //
	          & (parent_req     //
	             & (etree_req   //
	                | (post_req //
	                   & (postorder_req | colcount_req)))));
}

template <typename I>
void factorize_symbolic_non_zeros(
		SliceMut<I> nnz_per_col,
		SliceMut<I> etree,
		SliceMut<I> perm_inv,
		Slice<I> perm,
		SymbolicMatRef<I> a,
		DynStackMut stack) noexcept {
	{
		isize n = a.ncols();

		VEG_ASSERT_ALL_OF( //

				// perm non empty implies perm_inv non empty
				((perm_inv.len() == n || perm.len() == 0)),
				// perm[_inv] either empty or has size n
				((perm.len() == n || perm.len() == 0)),
				((perm_inv.len() == n || perm_inv.len() == 0)),

				a.nrows() == n,
				nnz_per_col.len() == n,
				etree.len() == n);
	}

	bool id_perm = perm_inv.len() == 0;
	bool user_perm = perm.len() != 0;

	Ordering o = user_perm ? Ordering::user_provided
	             : id_perm ? Ordering::natural
	                       : Ordering::amd;

	veg::Tag<I> tag{};

	usize n = usize(a.ncols());

	switch (o) {
	case Ordering::natural:
		break;

	case Ordering::amd: {
		auto amd_perm = stack.make_new_for_overwrite(tag, isize(n)).unwrap();
		sparse::amd(amd_perm.as_mut(), a, stack);
		perm = amd_perm.as_ref();
	}
		HEDLEY_FALL_THROUGH;
	case Ordering::user_provided: {
		_detail::inv_perm(perm_inv, perm);
	}
	default:
		break;
	}

	auto _permuted_a_col_ptrs =
			stack //
					.make_new_for_overwrite(tag, id_perm ? 0 : (a.ncols() + 1))
					.unwrap();
	auto _permuted_a_row_indices =
			stack //
					.make_new_for_overwrite(tag, id_perm ? 0 : (a.nnz()))
					.unwrap();

	if (!id_perm) {
		_permuted_a_col_ptrs.as_mut()[0] = 0;
		_permuted_a_col_ptrs.as_mut()[isize(n)] = I(a.nnz());
		SymbolicMatMut<I> permuted_a{
				from_raw_parts,
				isize(n),
				isize(n),
				a.nnz(),
				_permuted_a_col_ptrs.as_mut(),
				_permuted_a_row_indices.as_mut(),
				{},
		};
		_detail::symmetric_permute_symbolic(
				permuted_a, a, perm_inv.as_const(), stack);
	}

	SymbolicMatRef<I> permuted_a = id_perm ? a
	                                       : SymbolicMatRef<I>{
																							 from_raw_parts,
																							 isize(n),
																							 isize(n),
																							 a.nnz(),
																							 _permuted_a_col_ptrs.as_ref(),
																							 _permuted_a_row_indices.as_ref(),
																							 {},
																					 };

	sparse::etree(etree, permuted_a, stack);

	auto _post = stack.make_new_for_overwrite(tag, isize(n)).unwrap();
	sparse::postorder(_post.as_mut(), etree.as_const(), stack);

	sparse::column_counts(
			nnz_per_col, permuted_a, etree.as_const(), _post.as_ref(), stack);
}

template <typename I>
void factorize_symbolic_col_counts(
		SliceMut<I> col_ptrs,
		SliceMut<I> etree,
		SliceMut<I> perm_inv,
		Slice<I> perm,
		SymbolicMatRef<I> a,
		DynStackMut stack) noexcept {

	sparse::factorize_symbolic_non_zeros( //
			col_ptrs.split_at_mut(1)[1_c],
			etree,
			perm_inv,
			perm,
			a,
			stack);

	usize n = usize(a.ncols());
	auto pcol_ptrs = col_ptrs.ptr_mut();
	pcol_ptrs[0] = I(0);
	for (usize i = 0; i < n; ++i) {
		pcol_ptrs[i + 1] =
				util::checked_non_negative_plus(pcol_ptrs[i + 1], pcol_ptrs[i]);
	}
}

template <typename T, typename I>
auto factorize_numeric_req(
		veg::Tag<T> /*ttag*/,
		veg::Tag<I> /*itag*/,
		isize n,
		isize a_nnz,
		Ordering o) noexcept -> veg::dynstack::StackReq {
	using veg::dynstack::StackReq;

	constexpr isize sz{sizeof(I)};
	constexpr isize al{alignof(I)};

	constexpr isize tsz{sizeof(T)};
	constexpr isize tal{alignof(T)};

	bool id_perm = o == Ordering::natural;

	auto symb_perm_req = StackReq{sz * (id_perm ? 0 : (n + 1 + a_nnz)), al};
	auto num_perm_req = StackReq{tsz * (id_perm ? 0 : a_nnz), tal};
	return num_perm_req                       //
	       & (StackReq{tsz * n, tal}          //
	          & (symb_perm_req                //
	             & (StackReq{2 * n * sz, al}  //
	                & (StackReq{n * tsz, tal} //
	                   & StackReq{n * isize{sizeof(bool)}, alignof(bool)}))));
}

template <typename I>
void col_ptrs_to_nnz_per_col(
		SliceMut<I> nnz_per_col, Slice<I> col_ptrs) noexcept {
	VEG_ASSERT(nnz_per_col.len() + 1 == col_ptrs.len());
	isize n = usize(col_ptrs.len());

	I const* cp = col_ptrs.ptr();
	for (isize i = 0; i < n; ++i) {
		nnz_per_col.ptr_mut()[i] = I(cp[i + 1] - cp[i]);
	}
}

template <typename T, typename I>
void factorize_numeric( //
		T* values,
		I* row_indices,
		veg::DoNotDeduce<T const*> diag_to_add,
		veg::DoNotDeduce<I const*> perm,
		Slice<I> col_ptrs,
		Slice<I> etree,
		Slice<I> perm_inv,
		MatRef<T, I> a,
		DynStackMut stack) noexcept(false) {
	using namespace _detail;
	isize n = etree.len();
	VEG_ASSERT_ALL_OF( //
			col_ptrs.len() == n + 1,
			((perm_inv.len() == 0 || perm_inv.len() == n)),
			a.ncols() == n,
			a.nrows() == n);

	bool id_perm = perm_inv.len() == 0;

	veg::Tag<I> tag{};

	auto _permuted_a_values =
			stack.make_new_for_overwrite(veg::Tag<T>{}, id_perm ? 0 : a.nnz())
					.unwrap();

	auto _x = stack.make_new_for_overwrite(veg::Tag<T>{}, n).unwrap();

	auto _permuted_a_col_ptrs =
			stack.make_new_for_overwrite(tag, id_perm ? 0 : (a.ncols() + 1)).unwrap();
	auto _permuted_a_row_indices =
			stack.make_new_for_overwrite(tag, id_perm ? 0 : a.nnz()).unwrap();

	if (!id_perm) {
		_permuted_a_col_ptrs.as_mut()[0] = 0;
		_permuted_a_col_ptrs.as_mut()[n] = I(a.nnz());
		MatMut<T, I> permuted_a{
				from_raw_parts,
				n,
				n,
				a.nnz(),
				_permuted_a_col_ptrs.as_mut(),
				_permuted_a_row_indices.as_mut(),
				{},
				_permuted_a_values.as_mut(),
		};
		_detail::symmetric_permute(permuted_a, a, perm_inv, stack);
	}

	MatRef<T, I> permuted_a = id_perm ? a
	                                  : MatRef<T, I>{
																					from_raw_parts,
																					isize(n),
																					isize(n),
																					a.nnz(),
																					_permuted_a_col_ptrs.as_ref(),
																					_permuted_a_row_indices.as_ref(),
																					{},
																					_permuted_a_values.as_ref(),
																			};

	auto _current_row_index = stack.make_new_for_overwrite(tag, n).unwrap();
	auto _ereach_stack_storage = stack.make_new_for_overwrite(tag, n).unwrap();

	I* pcurrent_row_index = _current_row_index.as_mut().ptr_mut();
	T* px = _x.as_mut().ptr_mut();

	std::memcpy( //
			pcurrent_row_index,
			col_ptrs.ptr(),
			usize(n) * sizeof(I));
	for (usize i = 0; i < usize(n); ++i) {
		px[i] = 0;
	}

	// compute the iter-th row of L using the iter-th column of permuted_a
	// the diagonal element is filled with the diagonal of D instead of 1
	I const* plp = col_ptrs.ptr();

	auto _marked = stack.make_new(veg::Tag<bool>{}, n).unwrap();
	for (usize iter = 0; iter < usize(n); ++iter) {
		auto ereach_stack = _detail::ereach(
				_ereach_stack_storage.as_mut(),
				permuted_a.symbolic(),
				etree,
				isize(iter),
				_marked.ptr_mut());

		auto pereach_stack = ereach_stack.ptr();

		I const* pai = permuted_a.row_indices().ptr();
		T const* pax = permuted_a.values().ptr();

		I* pli = row_indices;
		T* plx = values;

		{
			auto col_start = permuted_a.col_start(iter);
			auto col_end = permuted_a.col_end(iter);

			// scatter permuted_a column into x
			// untouched columns are already zeroed

			for (usize p = col_start; p < col_end; ++p) {
				auto i = util::zero_extend(pai[p]);
				px[i] = pax[p];
			}
		}
		T d = px[iter] + ((diag_to_add == nullptr || perm == nullptr)
		                      ? T(0)
		                      : diag_to_add[util::zero_extend(perm[iter])]);

		// zero for next iteration
		px[iter] = 0;

		for (usize q = 0; q < usize(ereach_stack.len()); ++q) {
			usize j = util::zero_extend(pereach_stack[q]);
			auto col_start = util::zero_extend(plp[j]);
			auto row_idx = util::zero_extend(pcurrent_row_index[j]) + 1;

			T const xj = px[j];
			T const dj = plx[col_start];
			T const lkj = xj / dj;

			// zero for the next iteration
			px[j] = 0;

			// skip first element, to put diagonal there later
			for (usize p = col_start + 1; p < row_idx; ++p) {
				auto i = util::zero_extend(pli[p]);
				px[i] -= plx[p] * xj;
			}

			d -= lkj * xj;

			pli[row_idx] = I(iter);
			plx[row_idx] = lkj;
			pcurrent_row_index[j] = I(row_idx);
		}
		{
			auto col_start = util::zero_extend(plp[iter]);
			pli[col_start] = I(iter);
			plx[col_start] = d;
		}
	}
}
} // namespace sparse
} // namespace linearsolver
#endif /* end of include guard SPARSE_LDLT_FACTORIZE_HPP_P6FLZUBLS */
