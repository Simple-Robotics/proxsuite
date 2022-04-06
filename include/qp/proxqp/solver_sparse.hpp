#ifndef INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS
#define INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS

#include <dense-ldlt/core.hpp>
#include <sparse_ldlt/core.hpp>
#include <sparse_ldlt/factorize.hpp>
#include <sparse_ldlt/update.hpp>
#include <sparse_ldlt/rowmod.hpp>
#include <qp/views.hpp>
#include <qp/QPSettings.hpp>
#include <iostream>

namespace qp {
using veg::isize;

namespace sparse {
template <typename T, typename I>
struct QpView {
	sparse_ldlt::MatRef<T, I> H;
	sparse_ldlt::DenseVecRef<T> g;
	sparse_ldlt::MatRef<T, I> AT;
	sparse_ldlt::DenseVecRef<T> b;
	sparse_ldlt::MatRef<T, I> CT;
	sparse_ldlt::DenseVecRef<T> l;
	sparse_ldlt::DenseVecRef<T> u;
};

template <typename T, typename I>
struct QpViewMut {
	sparse_ldlt::MatMut<T, I> H;
	sparse_ldlt::DenseVecMut<T> g;
	sparse_ldlt::MatMut<T, I> AT;
	sparse_ldlt::DenseVecMut<T> b;
	sparse_ldlt::MatMut<T, I> CT;
	sparse_ldlt::DenseVecMut<T> l;
	sparse_ldlt::DenseVecMut<T> u;

	auto as_const() noexcept -> QpView<T, I> {
		return {
				H.as_const(),
				g.as_const(),
				AT.as_const(),
				b.as_const(),
				CT.as_const(),
				l.as_const(),
				u.as_const(),
		};
	}
};

namespace preconditioner {
enum struct Symmetry {
	LOWER,
	UPPER,
};

namespace detail {
template <typename T, typename I>
void rowwise_infty_norm(T* row_norm, sparse_ldlt::MatRef<T, I> m) {
	using namespace sparse_ldlt::util;

	I const* mi = m.row_indices().ptr();
	T const* mx = m.values().ptr();

	for (usize j = 0; j < usize(m.ncols()); ++j) {
		auto col_start = m.col_start(j);
		auto col_end = m.col_end(j);

		for (usize p = col_start; p < col_end; ++p) {
			usize i = zero_extend(mi[p]);
			T mij = fabs(mx[p]);
			row_norm[i] = std::max(row_norm[i], mij);
		}
	}
}

template <typename T, typename I>
void colwise_infty_norm_symhi(T* col_norm, sparse_ldlt::MatRef<T, I> h) {
	using namespace sparse_ldlt::util;

	I const* hi = h.row_indices().ptr();
	T const* hx = h.values().ptr();

	for (usize j = 0; j < usize(h.ncols()); ++j) {
		auto col_start = h.col_start(j);
		auto col_end = h.col_end(j);

		T norm_j = 0;

		for (usize p = col_start; p < col_end; ++p) {
			usize i = zero_extend(hi[p]);
			if (i > j) {
				break;
			}

			T hij = fabs(hx[p]);
			norm_j = std::max(norm_j, hij);
			col_norm[i] = std::max(col_norm[i], hij);
		}

		col_norm[j] = norm_j;
	}
}

template <typename T, typename I>
void colwise_infty_norm_symlo(T* col_norm, sparse_ldlt::MatRef<T, I> h) {
	using namespace sparse_ldlt::util;

	I const* hi = h.row_indices().ptr();
	T const* hx = h.values().ptr();

	for (usize j = 0; j < usize(h.ncols()); ++j) {
		auto col_start = h.col_start(j);
		auto col_end = h.col_end(j);

		T norm_j = 0;

		if (col_end > col_start) {
			usize p = col_end;
			while (true) {
				--p;
				usize i = zero_extend(hi[p]);
				if (i < j) {
					break;
				}

				T hij = fabs(hx[p]);
				norm_j = std::max(norm_j, hij);
				col_norm[i] = std::max(col_norm[i], hij);

				if (p <= col_start) {
					break;
				}
			}
		}
		col_norm[j] = std::max(col_norm[j], norm_j);
	}
}

template <typename T, typename I>
auto ruiz_scale_qp_in_place( //
		VectorViewMut<T> delta_,
		QpViewMut<T, I> qp,
		T epsilon,
		isize max_iter,
		Symmetry sym,
		veg::dynstack::DynStackMut stack) -> T {
	T c = 1;
	auto S = delta_.to_eigen();

	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();

	T gamma = 1;
	i64 iter = 1;

	LDLT_TEMP_VEC(T, delta, n + n_eq + n_in, stack);

	I* Hi = qp.H.row_indices_mut().ptr_mut();
	T* Hx = qp.H.values_mut().ptr_mut();

	I* ATi = qp.AT.row_indices_mut().ptr_mut();
	T* ATx = qp.AT.values_mut().ptr_mut();

	I* CTi = qp.CT.row_indices_mut().ptr_mut();
	T* CTx = qp.CT.values_mut().ptr_mut();

	T const machine_eps = std::numeric_limits<T>::epsilon();

	while (infty_norm((1 - delta.array()).matrix()) > epsilon) {
		if (iter == max_iter) {
			break;
		} else {
			++iter;
		}

		// norm_infty of each column of A (resp. C), i.e.,
		// each row of AT (resp. CT)
		{
			auto _a_infty_norm = stack.make_new(veg::Tag<T>{}, n).unwrap();
			auto _c_infty_norm = stack.make_new(veg::Tag<T>{}, n).unwrap();
			auto _h_infty_norm = stack.make_new(veg::Tag<T>{}, n).unwrap();
			T* a_infty_norm = _a_infty_norm.ptr_mut();
			T* c_infty_norm = _c_infty_norm.ptr_mut();
			T* h_infty_norm = _h_infty_norm.ptr_mut();

			detail::rowwise_infty_norm(a_infty_norm, qp.AT.as_const());
			detail::rowwise_infty_norm(c_infty_norm, qp.CT.as_const());
			switch (sym) {
			case Symmetry::LOWER: {
				detail::colwise_infty_norm_symlo(h_infty_norm, qp.H.as_const());
				break;
			}
			case Symmetry::UPPER: {
				detail::colwise_infty_norm_symhi(h_infty_norm, qp.H.as_const());
				break;
			}
			}

			for (isize j = 0; j < n; ++j) {
				delta(j) = T(1) / (machine_eps + sqrt(std::max({
																						 h_infty_norm[j],
																						 a_infty_norm[j],
																						 c_infty_norm[j],
																				 })));
			}
		}
		using namespace sparse_ldlt::util;
		for (usize j = 0; j < usize(n_eq); ++j) {
			T a_row_norm = 0;
			qp.AT.to_eigen();
			usize col_start = qp.AT.col_start(j);
			usize col_end = qp.AT.col_end(j);

			for (usize p = col_start; p < col_end; ++p) {
				T aji = fabs(ATx[p]);
				a_row_norm = std::max(a_row_norm, aji);
			}

			delta(n + isize(j)) = T(1) / (machine_eps + sqrt(a_row_norm));
		}

		for (usize j = 0; j < usize(n_in); ++j) {
			T c_row_norm = 0;
			usize col_start = qp.CT.col_start(j);
			usize col_end = qp.CT.col_end(j);

			for (usize p = col_start; p < col_end; ++p) {
				T cji = fabs(CTx[p]);
				c_row_norm = std::max(c_row_norm, cji);
			}

			delta(n + n_eq + isize(j)) = T(1) / (machine_eps + sqrt(c_row_norm));
		}

		// normalize A
		for (usize j = 0; j < usize(n_eq); ++j) {
			usize col_start = qp.AT.col_start(j);
			usize col_end = qp.AT.col_end(j);

			T delta_j = delta(n + isize(j));

			for (usize p = col_start; p < col_end; ++p) {
				usize i = zero_extend(ATi[p]);
				T& aji = ATx[p];
				T delta_i = delta(isize(i));
				aji = delta_i * (aji * delta_j);
			}
		}

		// normalize C
		for (usize j = 0; j < usize(n_in); ++j) {
			usize col_start = qp.CT.col_start(j);
			usize col_end = qp.CT.col_end(j);

			T delta_j = delta(n + n_eq + isize(j));

			for (usize p = col_start; p < col_end; ++p) {
				usize i = zero_extend(CTi[p]);
				T& cji = CTx[p];
				T delta_i = delta(isize(i));
				cji = delta_i * (cji * delta_j);
			}
		}

		// normalize H
		switch (sym) {
		case Symmetry::LOWER: {
			for (usize j = 0; j < usize(n); ++j) {
				usize col_start = qp.H.col_start(j);
				usize col_end = qp.H.col_end(j);
				T delta_j = delta(isize(j));

				if (col_end > col_start) {
					usize p = col_end;
					while (true) {
						--p;
						usize i = zero_extend(Hi[p]);
						if (i < j) {
							break;
						}
						Hx[p] = delta_j * Hx[p] * delta(isize(i));

						if (p <= col_start) {
							break;
						}
					}
				}
			}
			break;
		}
		case Symmetry::UPPER: {
			for (usize j = 0; j < usize(n); ++j) {
				usize col_start = qp.H.col_start(j);
				usize col_end = qp.H.col_end(j);
				T delta_j = delta(isize(j));

				for (usize p = col_start; p < col_end; ++p) {
					usize i = zero_extend(Hi[p]);
					if (i > j) {
						break;
					}
					Hx[p] = delta_j * Hx[p] * delta(isize(i));
				}
			}
			break;
		}
		}

		// normalize vectors
		qp.g.to_eigen().array() *= delta.head(n).array();
		qp.b.to_eigen().array() *= delta.segment(n, n_eq).array();
		qp.l.to_eigen().array() *= delta.tail(n_in).array();
		qp.u.to_eigen().array() *= delta.tail(n_in).array();

		// additional normalization
		auto _h_infty_norm = stack.make_new(veg::Tag<T>{}, n).unwrap();
		T* h_infty_norm = _h_infty_norm.ptr_mut();

		switch (sym) {
		case Symmetry::LOWER: {
			detail::colwise_infty_norm_symlo(h_infty_norm, qp.H.as_const());
			break;
		}
		case Symmetry::UPPER: {
			detail::colwise_infty_norm_symhi(h_infty_norm, qp.H.as_const());
			break;
		}
		}

		T avg = 0;
		for (isize i = 0; i < n; ++i) {
			avg += h_infty_norm[i];
		}
		avg /= T(n);

		gamma = 1 / std::max(avg, T(1));

		qp.g.to_eigen() *= gamma;
		qp.H.to_eigen() *= gamma;

		S.array() *= delta.array();
		c *= gamma;
	}
	return c;
}
} // namespace detail

template <typename T, typename I>
struct RuizEquilibration {
	Eigen::Matrix<T, -1, 1> delta;
	isize n;
	T c;
	T epsilon;
	i64 max_iter;
	Symmetry sym;

	std::ostream* logger_ptr = nullptr;

	RuizEquilibration(
			isize n_,
			isize n_eq_in,
			T epsilon_ = T(1e-3),
			i64 max_iter_ = 10,
			Symmetry sym_ = Symmetry::UPPER,
			std::ostream* logger = nullptr)
			: delta(Eigen::Matrix<T, -1, 1>::Ones(n_ + n_eq_in)),
				n(n_),
				c(1),
				epsilon(epsilon_),
				max_iter(max_iter_),
				sym(sym_),
				logger_ptr(logger) {}

	static auto
	scale_qp_in_place_req(veg::Tag<T> tag, isize n, isize n_eq, isize n_in)
			-> veg::dynstack::StackReq {
		return dense_ldlt::temp_vec_req(tag, n + n_eq + n_in) &
		       veg::dynstack::StackReq::with_len(tag, 3 * n);
	}

	void scale_qp_in_place(QpViewMut<T, I> qp, veg::dynstack::DynStackMut stack) {
		delta.setOnes();
		c = detail::ruiz_scale_qp_in_place( //
				{ldlt::from_eigen, delta},
				qp,
				epsilon,
				max_iter,
				sym,
				stack);
	}

	// modifies variables in place
	void scale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.array().head(n);
	}
	void scale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() /
		                          delta.tail(delta.size() - n).array() * c;
	}

	void scale_dual_in_place_eq(VectorViewMut<T> dual) {
		dual.to_eigen().array() =
				dual.as_const().to_eigen().array() /
				delta.middleRows(n, dual.to_eigen().size()).array() * c;
	}
	void scale_dual_in_place_in(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() /
		                          delta.tail(dual.to_eigen().size()).array() * c;
	}

	void unscale_primal_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.array().head(n);
	}
	void unscale_dual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() *
		                          delta.tail(delta.size() - n).array() / c;
	}

	void unscale_dual_in_place_eq(VectorViewMut<T> dual) {
		dual.to_eigen().array() =
				dual.as_const().to_eigen().array() *
				delta.middleRows(n, dual.to_eigen().size()).array() / c;
	}

	void unscale_dual_in_place_in(VectorViewMut<T> dual) {
		dual.to_eigen().array() = dual.as_const().to_eigen().array() *
		                          delta.tail(dual.to_eigen().size()).array() / c;
	}
	// modifies residuals in place
	void scale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() *= delta.tail(delta.size() - n).array();
	}

	void scale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) {
		primal_eq.to_eigen().array() *=
				delta.middleRows(n, primal_eq.to_eigen().size()).array();
	}
	void scale_primal_residual_in_place_in(VectorViewMut<T> primal_in) {
		primal_in.to_eigen().array() *=
				delta.tail(primal_in.to_eigen().size()).array();
	}
	void scale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() *= delta.head(n).array() * c;
	}
	void unscale_primal_residual_in_place(VectorViewMut<T> primal) {
		primal.to_eigen().array() /= delta.tail(delta.size() - n).array();
	}
	void unscale_primal_residual_in_place_eq(VectorViewMut<T> primal_eq) {
		primal_eq.to_eigen().array() /=
				delta.middleRows(n, primal_eq.to_eigen().size()).array();
	}
	void unscale_primal_residual_in_place_in(VectorViewMut<T> primal_in) {
		primal_in.to_eigen().array() /=
				delta.tail(primal_in.to_eigen().size()).array();
	}
	void unscale_dual_residual_in_place(VectorViewMut<T> dual) {
		dual.to_eigen().array() /= delta.head(n).array() * c;
	}
};

} // namespace preconditioner

template <typename T, typename I>
using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T, typename I>
struct QpWorkspace {

	struct {
		veg::Vec<veg::mem::byte> storage;
		veg::Vec<I> kkt_col_ptrs;
		veg::Vec<I> kkt_row_indices;
		veg::Vec<T> kkt_values;

		veg::Vec<I> ldl_col_ptrs;
		veg::Vec<I> perm_inv;

		auto stack_mut() -> veg::dynstack::DynStackMut {
			return {
					veg::from_slice_mut,
					storage.as_mut(),
			};
		}

		void setup_impl(QpView<T, I> qp, veg::dynstack::StackReq precond_req) {
			using namespace veg::dynstack;
			using namespace sparse_ldlt::util;

			using SR = StackReq;
			veg::Tag<I> itag;
			veg::Tag<T> xtag;

			isize n = qp.H.nrows();
			isize n_eq = qp.AT.ncols();
			isize n_in = qp.CT.ncols();
			isize n_tot = n + n_eq + n_in;

			isize nnz_tot = qp.H.nnz() + qp.AT.nnz() + qp.CT.nnz();

			// form the full kkt matrix
			// assuming H, AT, CT are sorted
			// and H is upper triangular
			{
				kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
				kkt_row_indices.resize_for_overwrite(nnz_tot);
				kkt_values.resize_for_overwrite(nnz_tot);

				I* kktp = kkt_col_ptrs.ptr_mut();
				I* kkti = kkt_row_indices.ptr_mut();
				T* kktx = kkt_values.ptr_mut();

				kktp[0] = 0;
				usize col = 0;
				usize pos = 0;

				auto insert_submatrix = [&](sparse_ldlt::MatRef<T, I> m,
				                            bool assert_sym_hi) -> void {
					I const* mi = m.row_indices().ptr();
					T const* mx = m.values().ptr();
					isize ncols = m.ncols();

					for (usize j = 0; j < usize(ncols); ++j) {
						usize col_start = m.col_start(j);
						usize col_end = m.col_end(j);

						kktp[col + 1] =
								checked_non_negative_plus(kktp[col], I(col_end - col_start));
						++col;

						for (usize p = col_start; p < col_end; ++p) {
							usize i = zero_extend(mi[p]);
							if (assert_sym_hi) {
								VEG_ASSERT(i <= j);
							}

							kkti[pos] = veg::nb::narrow<I>{}(i);
							kktx[pos] = mx[p];

							++pos;
						}
					}
				};

				insert_submatrix(qp.H, true);
				insert_submatrix(qp.AT, false);
				insert_submatrix(qp.CT, false);
			}

			storage.resize_for_overwrite( //
					(StackReq::with_len(itag, n_tot) &
			     sparse_ldlt::factorize_symbolic_req( //
							 itag,                            //
							 n_tot,                           //
							 nnz_tot,                         //
							 sparse_ldlt::Ordering::amd))     //
							.alloc_req()                      //
			);

			ldl_col_ptrs.resize_for_overwrite(n_tot + 1);
			perm_inv.resize_for_overwrite(n_tot);

			DynStackMut stack = stack_mut();

			{
				auto _etree = stack.make_new_for_overwrite(itag, n_tot).unwrap();
				auto etree = _etree.as_mut();

				sparse_ldlt::factorize_symbolic_col_counts( //
						ldl_col_ptrs.as_mut(),
						etree,
						perm_inv.as_mut(),
						{},
						{
								sparse_ldlt::from_raw_parts,
								n_tot,
								n_tot,
								nnz_tot,
								kkt_col_ptrs.as_ref(),
								kkt_row_indices.as_ref(),
								{},
						},
						stack);
			}

			auto lnnz = isize(zero_extend(ldl_col_ptrs[n_tot]));
			auto req = SR::and_(veg::init_list({
										 dense_ldlt::temp_vec_req(veg::Tag<T>{}, n),    // g_scaled
										 dense_ldlt::temp_vec_req(veg::Tag<T>{}, n_eq), // b_scaled
										 dense_ldlt::temp_vec_req(veg::Tag<T>{}, n_in), // l_scaled
										 dense_ldlt::temp_vec_req(veg::Tag<T>{}, n_in), // u_scaled
								 })) &
			           (precond_req | // scale_qp_in_place
			            (SR::and_(veg::init_list({
											 SR::with_len(itag, n_tot), // perm
											 SR::with_len(itag, n_tot), // etree
											 SR::with_len(itag, n_tot), // ldl nnz counts
											 SR::with_len(itag, lnnz),  // ldl row indices
											 SR::with_len(xtag, lnnz),  // ldl values
									 })) &
			             (sparse_ldlt::factorize_symbolic_req( // symbolic ldl
												itag,
												n_tot,
												nnz_tot,
												sparse_ldlt::Ordering::user_provided) |
			              (SR::with_len(xtag, n_tot) &         // diag
			               sparse_ldlt::factorize_numeric_req( // numeric ldl
												 xtag,
												 itag,
												 n_tot,
												 nnz_tot,
												 sparse_ldlt::Ordering::user_provided)))));

			storage.resize_for_overwrite(req.alloc_req());
		}
	} _;

	QpWorkspace() = default;

	auto ldl_col_ptrs() const -> veg::Slice<I> { return _.ldl_col_ptrs.as_ref(); }
	auto ldl_col_ptrs_mut() -> veg::SliceMut<I> {
		return _.ldl_col_ptrs.as_mut();
	}
	auto stack_mut() -> veg::dynstack::DynStackMut { return _.stack_mut(); }

	auto kkt() const -> sparse_ldlt::MatMut<T, I> {
		auto n_tot = _.kkt_col_ptrs.len() - 1;
		auto nnz = isize(sparse_ldlt::util::zero_extend(_.kkt_col_ptrs[n_tot]));
		return {
				sparse_ldlt::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				_.kkt_col_ptrs.as_ref(),
				_.kkt_row_indices.as_ref(),
				{},
				_.kkt_values.as_ref(),
		};
	}
	auto kkt_mut() -> sparse_ldlt::MatMut<T, I> {
		auto n_tot = _.kkt_col_ptrs.len() - 1;
		auto nnz = isize(sparse_ldlt::util::zero_extend(_.kkt_col_ptrs[n_tot]));
		return {
				sparse_ldlt::from_raw_parts,
				n_tot,
				n_tot,
				nnz,
				_.kkt_col_ptrs.as_mut(),
				_.kkt_row_indices.as_mut(),
				{},
				_.kkt_values.as_mut(),
		};
	}
};

template <typename T, typename I, typename P>
void qp_setup(QpWorkspace<T, I>& work, QpView<T, I> qp, P& precond) {
	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();
	work._.setup_impl(qp, P::scale_qp_in_place_req(veg::Tag<T>{}, n, n_eq, n_in));
}

template <typename T, typename I, typename P>
void qp_solve(
		VectorViewMut<T> x,
		VectorViewMut<T> y,
		VectorViewMut<T> z,
		QpWorkspace<T, I>& work,
		QPSettings<T> const& settings,
		P& precond,
		QpView<T, I> qp) {

	using namespace sparse_ldlt::tags;
	using namespace veg::literals;
	namespace util = sparse_ldlt::util;

	veg::dynstack::DynStackMut stack = work.stack_mut();

	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();
	isize n_tot = n + n_eq + n_in;

	sparse_ldlt::MatMut<T, I> kkt = work.kkt_mut();

	sparse_ldlt::MatMut<T, I> H_scaled = {
			sparse_ldlt::from_raw_parts,
			n,
			n,
			qp.H.nnz(),
			kkt.col_ptrs_mut().split_at_mut(n + 1)[0_c],
			kkt.row_indices_mut(),
			{},
			kkt.values_mut(),
	};

	sparse_ldlt::MatMut<T, I> AT_scaled = {
			sparse_ldlt::from_raw_parts,
			n,
			n_eq,
			qp.AT.nnz(),
			kkt.col_ptrs_mut() //
					.split_at_mut(n)[1_c]
					.split_at_mut(n_eq + 1)[0_c],
			kkt.row_indices_mut(),
			{},
			kkt.values_mut(),
	};

	sparse_ldlt::MatMut<T, I> CT_scaled = {
			sparse_ldlt::from_raw_parts,
			n,
			n_in,
			qp.CT.nnz(),
			kkt.col_ptrs_mut() //
					.split_at_mut(n + n_eq)[1_c]
					.split_at_mut(n_in + 1)[0_c],
			kkt.row_indices_mut(),
			{},
			kkt.values_mut(),
	};

	auto H_scaled_e = H_scaled.to_eigen();
	LDLT_TEMP_VEC_UNINIT(T, g_scaled_e, n, stack);
	auto AT_scaled_e = AT_scaled.to_eigen();
	LDLT_TEMP_VEC_UNINIT(T, b_scaled_e, n_eq, stack);
	auto CT_scaled_e = CT_scaled.to_eigen();
	LDLT_TEMP_VEC_UNINIT(T, l_scaled_e, n_in, stack);
	LDLT_TEMP_VEC_UNINIT(T, u_scaled_e, n_in, stack);

	g_scaled_e = qp.g.to_eigen();
	b_scaled_e = qp.b.to_eigen();
	l_scaled_e = qp.l.to_eigen();
	u_scaled_e = qp.u.to_eigen();

	QpViewMut<T, I> qp_scaled = {
			H_scaled,
			{from_eigen, g_scaled_e},
			AT_scaled,
			{from_eigen, b_scaled_e},
			CT_scaled,
			{from_eigen, l_scaled_e},
			{from_eigen, u_scaled_e},
	};

	precond.scale_qp_in_place(qp_scaled, stack);

	T primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	auto ldl_col_ptrs = work.ldl_col_ptrs();
	auto max_lnnz = isize(util::zero_extend(ldl_col_ptrs[n_tot]));

	veg::Tag<I> itag;
	veg::Tag<T> xtag;

	auto _perm = stack.make_new_for_overwrite(itag, n_tot).unwrap();
	auto _etree = stack.make_new_for_overwrite(itag, n_tot).unwrap();
	auto _ldl_nnz_counts = stack.make_new_for_overwrite(itag, n_tot).unwrap();
	auto _ldl_row_indices = stack.make_new_for_overwrite(itag, max_lnnz).unwrap();
	auto _ldl_values = stack.make_new_for_overwrite(xtag, max_lnnz).unwrap();

	veg::Slice<I> perm_inv = work._.perm_inv.as_ref();
	veg::SliceMut<I> perm = _etree.as_mut();
	for (isize i = 0; i < n_tot; ++i) {
		perm[util::zero_extend(perm_inv[i])];
	}

	veg::SliceMut<I> etree = _etree.as_mut();
	veg::SliceMut<I> ldl_nnz_counts = _ldl_nnz_counts.as_mut();
	veg::SliceMut<I> ldl_row_indices = _ldl_row_indices.as_mut();
	veg::SliceMut<T> ldl_values = _ldl_values.as_mut();

	{
    // FIXME:
    // use only active kkt columns
    // store kkt in uncompressed format
		sparse_ldlt::factorize_symbolic_non_zeros(
				ldl_nnz_counts,
				etree,
				work._.perm_inv.as_mut(),
				perm,
				kkt.symbolic(),
				stack);

		auto _diag = stack.make_new_for_overwrite(xtag, n_tot).unwrap();
		T* diag = _diag.ptr_mut();
	}
}
} // namespace sparse
} // namespace qp

#endif /* end of include guard INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS */
