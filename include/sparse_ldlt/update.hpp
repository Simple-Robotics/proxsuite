#ifndef SPARSE_LDLT_UPDATE_HPP_T3WZ0HOXS
#define SPARSE_LDLT_UPDATE_HPP_T3WZ0HOXS

#include "sparse_ldlt/core.hpp"
#include <veg/tuple.hpp>
#include <algorithm>

namespace sparse_ldlt {
template <typename I>
auto merge_second_col_into_first_req(
		veg::Tag<I> /*tag*/, isize second_size) noexcept
		-> veg::dynstack::StackReq {
	return {
			second_size * isize{sizeof(I)},
			alignof(I),
	};
}

template <typename T, typename I>
auto merge_second_col_into_first( //
		I* difference,
		T* first_values,
		I* first_ptr,
		isize first_full_len,
		isize first_initial_len,
		Slice<I> second,
		veg::DoNotDeduce<I> ignore_threshold_inclusive,
		bool move_values,
		DynStackMut stack) noexcept(false)
		-> veg::Tuple<SliceMut<T>, SliceMut<I>, SliceMut<I>> {
	VEG_CHECK_CONCEPT(trivially_copyable<I>);
	VEG_CHECK_CONCEPT(trivially_copyable<T>);

	if (second.len() == 0) {
		return {
				veg::tuplify,
				{unsafe, from_raw_parts, first_values, first_initial_len},
				{unsafe, from_raw_parts, first_ptr, first_initial_len},
				{unsafe, from_raw_parts, difference, 0},
		};
	}

	I const* second_ptr = second.ptr();
	usize second_len = usize(second.len());

	usize index_second = 0;

	for (; index_second < second_len; ++index_second) {
		if (second_ptr[index_second] > ignore_threshold_inclusive) {
			break;
		}
	}
	auto ufirst_initial_len = usize(first_initial_len);

	second_ptr += index_second;
	second_len -= index_second;
	index_second = 0;

	veg::Tag<I> tag{};

	auto _ins_pos = stack.make_new_for_overwrite(tag, isize(second_len)).unwrap();

	I* insert_pos_ptr = _ins_pos.ptr_mut();
	usize insert_count = 0;

	for (usize index_first = 0; index_first < ufirst_initial_len; ++index_first) {
		I current_first = first_ptr[index_first];
		while (true) {
			if (!(index_second < second_len)) {
				break;
			}

			I current_second = second_ptr[index_second];
			if (!(current_second < current_first)) {
				break;
			}

			insert_pos_ptr[insert_count] = index_first;
			difference[insert_count] = current_second;
			++insert_count;
			++index_second;
		}

		if (index_second == second_len) {
			break;
		}
		if (second_ptr[index_second] == current_first) {
			++index_second;
		}
	}

	usize remaining_insert_count = insert_count;
	usize first_new_len =
			ufirst_initial_len + insert_count + (second_len - index_second);
	VEG_ASSERT(first_full_len >= first_new_len);

	usize append_count = second_len - index_second;
	std::memmove( //
			difference + insert_count,
			second_ptr + index_second,
			append_count * sizeof(I));
	std::memmove( //
			first_ptr + (ufirst_initial_len + insert_count),
			second_ptr + index_second,
			append_count * sizeof(I));
	if (move_values) {
		for (usize i = 0; i < append_count; ++i) {
			first_values[i + ufirst_initial_len + insert_count] = 0;
		}
	}

	while (remaining_insert_count != 0) {

		usize old_insert_pos = usize(insert_pos_ptr[remaining_insert_count - 1]);
		usize range_size =
				(remaining_insert_count == insert_count)
						? ufirst_initial_len - old_insert_pos
						: usize(insert_pos_ptr[remaining_insert_count]) - old_insert_pos;

		usize old_pos = old_insert_pos;
		usize new_pos = old_pos + remaining_insert_count;

		std::memmove( //
				first_ptr + new_pos,
				first_ptr + old_pos,
				range_size * sizeof(I));
		if (move_values) {
			std::memmove( //
					first_values + new_pos,
					first_values + old_pos,
					range_size * sizeof(T));
			first_values[new_pos - 1] = 0;
		}

		first_ptr[new_pos - 1] = difference[remaining_insert_count - 1];
		--remaining_insert_count;
	}

	return {
			veg::tuplify,
			{unsafe, from_raw_parts, first_values, isize(first_new_len)},
			{unsafe, from_raw_parts, first_ptr, isize(first_new_len)},
			{unsafe, from_raw_parts, difference, isize(insert_count + append_count)},
	};
}

template <typename T, typename I>
auto rank1_update_req( //
		veg::Tag<T> /*tag*/,
		veg::Tag<I> /*tag*/,
		isize n,
		bool id_perm,
		isize col_nnz) noexcept -> veg::dynstack::StackReq {
	using veg::dynstack::StackReq;
	StackReq permuted_indices = {
			id_perm ? 0 : (col_nnz * isize{sizeof(I)}), isize{alignof(I)}};
	StackReq difference = {n * isize{sizeof(I)}, isize{alignof(I)}};
	difference = difference & difference;

	StackReq merge =
			sparse_ldlt::merge_second_col_into_first_req(veg::Tag<I>{}, n);

	StackReq numerical_workspace = {n * isize{sizeof(T)}, isize{alignof(T)}};

	return permuted_indices & ((difference & merge) | numerical_workspace);
}

template <typename T, typename I>
auto rank1_update(
		MatMut<T, I> ld,
		SliceMut<I> etree,
		Slice<I> perm_inv,
		VecRef<T, I> w,
		veg::DoNotDeduce<T> alpha,
		DynStackMut stack) noexcept(false) -> MatMut<T, I> {
	VEG_ASSERT(!ld.is_compressed());

	if (w.nnz() == 0) {
		return ld;
	}

	veg::Tag<I> tag;
	usize n = usize(ld.ncols());
	bool id_perm = perm_inv.len() == 0;

	auto _w_permuted_indices =
			stack.make_new_for_overwrite(tag, id_perm ? isize(0) : w.nnz()).unwrap();

	auto w_permuted_indices =
			id_perm ? w.row_indices() : _w_permuted_indices.as_ref();
	if (!id_perm) {
		I* pw_permuted_indices = _w_permuted_indices.ptr_mut();
		for (usize k = 0; k < w.nnz(); ++k) {
			usize i = util::zero_extend(w.row_indices().ptr()[k]);
			pw_permuted_indices[k] = perm_inv.ptr()[i];
		}
		std::sort(pw_permuted_indices, pw_permuted_indices + w.nnz());
	}

	auto sx = util::sign_extend;
	// symbolic update
	{
		usize current_col = w_permuted_indices[0];

		auto _difference =
				stack.make_new_for_overwrite(tag, isize(n - current_col)).unwrap();
		auto _difference_backup =
				stack.make_new_for_overwrite(tag, isize(n - current_col)).unwrap();

		auto merge_col = w_permuted_indices;
		I* difference = _difference.ptr_mut();

		while (true) {
			usize old_parent = sx(etree[current_col]);

			usize current_ptr_idx = util::zero_extend(ld.col_ptrs()[current_col]);
			usize next_ptr_idx = util::zero_extend(ld.col_ptrs()[current_col + 1]);

			VEG_BIND(
					auto,
					(_, new_current_col, computed_difference),
					sparse_ldlt::merge_second_col_into_first(
							difference,
							ld.values_mut().ptr_mut() + (current_ptr_idx + 1),
							ld.row_indices_mut().ptr_mut() + (current_ptr_idx + 1),
							next_ptr_idx - current_ptr_idx,
							ld.nnz_per_col()[current_col] - 1,
							merge_col,
							current_col,
							true,
							stack));

			(void)_;
			ld._set_nnz(
					ld.nnz() + new_current_col.len() + 1 -
					isize(ld.nnz_per_col()[current_col]));
			ld.nnz_per_col_mut()[current_col] = I(new_current_col.len() + 1);

			usize new_parent =
					(new_current_col.len() == 0) ? usize(-1) : new_current_col[0];

			if (new_parent == usize(-1)) {
				break;
			}

			if (new_parent == old_parent) {
				merge_col = computed_difference.as_const();
				difference = _difference_backup.ptr_mut();
			} else {
				merge_col = new_current_col.as_const();
				difference = _difference.ptr_mut();
				etree[current_col] = I(new_parent);
			}

			current_col = new_parent;
		}
	}

	// numerical update
	{
		usize first_col = w_permuted_indices[0];
		auto _work = stack.make_new_for_overwrite(veg::Tag<T>{}, n).unwrap();
		T* pwork = _work.ptr_mut();

		for (usize col = first_col; col != usize(-1); col = sx(etree[col])) {
			pwork[col] = 0;
		}
		for (usize p = 0; p < w.nnz(); ++p) {
			pwork
					[id_perm ? w.row_indices()[p]
			             : (util::zero_extend(perm_inv[w.row_indices()[p]]))] =
							w.values()[p];
		}

		I const* pldi = ld.row_indices().ptr();
		T* pldx = ld.values_mut().ptr_mut();

		for (usize col = first_col; col != usize(-1); col = sx(etree[col])) {
			auto col_start = ld.col_start(col);
			auto col_end = ld.col_end(col);

			T w0 = pwork[col];
			T old_d = pldx[col_start];
			T new_d = old_d + alpha * w0 * w0;
			T beta = alpha * w0 / new_d;
			alpha = alpha - new_d * beta * beta;

			pldx[col_start] = new_d;
			pwork[col] -= w0;

			for (usize p = col_start + 1; p < col_end; ++p) {
				usize i = util::zero_extend(pldi[p]);

				T tmp = pldx[p];
				pwork[i] = pwork[i] - w0 * tmp;
				pldx[p] = tmp + beta * pwork[i];
			}
		}
	}

	return ld;
}
} // namespace sparse_ldlt

#endif /* end of include guard SPARSE_LDLT_UPDATE_HPP_T3WZ0HOXS */
