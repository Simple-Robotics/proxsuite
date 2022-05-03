#ifndef INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS
#define INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS

#include <dense-ldlt/core.hpp>
#include <sparse_ldlt/core.hpp>
#include <sparse_ldlt/factorize.hpp>
#include <sparse_ldlt/update.hpp>
#include <sparse_ldlt/rowmod.hpp>
#include <qp/dense/views.hpp>
#include <qp/Settings.hpp>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace qp {
namespace sparse {
using veg::isize;
using veg::usize;
using veg::i64;
using dense::VectorView;
using dense::VectorViewMut;
using dense::infty_norm;

template <typename T>
struct PrimalDualGradResult {
	T a;
	T b;
	T grad;
	VEG_REFLECT(PrimalDualGradResult, a, b, grad);
};

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

template <typename T, typename I>
struct Identity {

	static auto scale_qp_in_place_req(
			veg::Tag<T> /*tag*/, isize /*n*/, isize /*n_eq*/, isize /*n_in*/)
			-> veg::dynstack::StackReq {
		return {0, 1};
	}

	void scale_qp_in_place(
			QpViewMut<T, I> /*qp*/, veg::dynstack::DynStackMut /*stack*/) {}

	// modifies variables in place
	void scale_primal_in_place(VectorViewMut<T> /*primal*/) {}
	void scale_dual_in_place(VectorViewMut<T> /*dual*/) {}

	void scale_dual_in_place_eq(VectorViewMut<T> /*dual*/) {}
	void scale_dual_in_place_in(VectorViewMut<T> /*dual*/) {}

	void unscale_primal_in_place(VectorViewMut<T> /*primal*/) {}
	void unscale_dual_in_place(VectorViewMut<T> /*dual*/) {}

	void unscale_dual_in_place_eq(VectorViewMut<T> /*dual*/) {}

	void unscale_dual_in_place_in(VectorViewMut<T> /*dual*/) {}
	// modifies residuals in place
	void scale_primal_residual_in_place(VectorViewMut<T> /*primal*/) {}

	void scale_primal_residual_in_place_eq(VectorViewMut<T> /*primal_eq*/) {}
	void scale_primal_residual_in_place_in(VectorViewMut<T> /*primal_in*/) {}
	void scale_dual_residual_in_place(VectorViewMut<T> /*dual*/) {}
	void unscale_primal_residual_in_place(VectorViewMut<T> /*primal*/) {}
	void unscale_primal_residual_in_place_eq(VectorViewMut<T> /*primal_eq*/) {}
	void unscale_primal_residual_in_place_in(VectorViewMut<T> /*primal_in*/) {}
	void unscale_dual_residual_in_place(VectorViewMut<T> /*dual*/) {}
};

} // namespace preconditioner

template <typename T, typename I>
using Mat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template <typename T>
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T, typename I>
struct QpWorkspace {

	struct /* NOLINT */ {
		veg::Vec<veg::mem::byte> storage;
		veg::Vec<I> kkt_col_ptrs;
		veg::Vec<I> kkt_row_indices;
		veg::Vec<T> kkt_values;

		veg::Vec<I> ldl_col_ptrs;
		veg::Vec<I> perm_inv;
		bool do_ldlt;

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

			bool overflow = false;
			{
				auto _etree = stack.make_new_for_overwrite(itag, n_tot).unwrap();
				auto etree = _etree.as_mut();

				using namespace veg::literals;
				auto kkt_sym = sparse_ldlt::SymbolicMatRef<I>{
						sparse_ldlt::from_raw_parts,
						n_tot,
						n_tot,
						nnz_tot,
						kkt_col_ptrs.as_ref(),
						kkt_row_indices.as_ref(),
						{},
				};
				sparse_ldlt::factorize_symbolic_non_zeros( //
						ldl_col_ptrs.as_mut().split_at_mut(1)[1_c],
						etree,
						perm_inv.as_mut(),
						{},
						kkt_sym,
						stack);

				auto pcol_ptrs = ldl_col_ptrs.as_mut().ptr_mut();
				pcol_ptrs[0] = I(0);

				using veg::u64;
				u64 acc = 0;

				for (usize i = 0; i < usize(n_tot); ++i) {
					acc += u64(zero_extend(pcol_ptrs[i + 1]));
					if (acc != u64(I(acc))) {
						overflow = true;
					}
					pcol_ptrs[(i + 1)] = I(acc);
				}
			}

			auto lnnz = isize(zero_extend(ldl_col_ptrs[n_tot]));

			// if ldlt much sparser than kkt
			// do_ldlt = !overflow && lnnz < (100 * nnz_tot);
			do_ldlt = !overflow && lnnz < 10000000;

#define PROX_QP_ALL_OF(...)                                                    \
	::veg::dynstack::StackReq::and_(::veg::init_list(__VA_ARGS__))
#define PROX_QP_ANY_OF(...)                                                    \
	::veg::dynstack::StackReq::or_(::veg::init_list(__VA_ARGS__))

			auto refactorize_req =
					do_ldlt ? PROX_QP_ANY_OF({
												sparse_ldlt::factorize_symbolic_req( // symbolic ldl
														itag,
														n_tot,
														nnz_tot,
														sparse_ldlt::Ordering::user_provided),
												PROX_QP_ALL_OF({
														SR::with_len(xtag, n_tot),          // diag
														sparse_ldlt::factorize_numeric_req( // numeric ldl
																xtag,
																itag,
																n_tot,
																nnz_tot,
																sparse_ldlt::Ordering::user_provided),
												}),
										})
									: PROX_QP_ALL_OF({
												SR::with_len(itag, 0),
												SR::with_len(xtag, 0),
										});

			auto x_vec = [&](isize n) noexcept -> StackReq {
				return dense_ldlt::temp_vec_req(xtag, n);
			};

			auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // tmp
					x_vec(n_tot), // err
					x_vec(n_tot), // work
			});

			auto unscaled_primal_dual_residual_req = x_vec(n); // Hx
			auto line_search_req = PROX_QP_ALL_OF({
					x_vec(2 * n_in), // alphas
					x_vec(n),        // Cdx_active
					x_vec(n_in),     // active_part_z
					x_vec(n_in),     // tmp_lo
					x_vec(n_in),     // tmp_up
			});
			auto primal_dual_newton_semi_smooth_req = PROX_QP_ALL_OF({
					x_vec(n_tot), // dw
					PROX_QP_ANY_OF({
							ldl_solve_in_place_req,
							PROX_QP_ALL_OF({
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_lo
									SR::with_len(veg::Tag<bool>{}, n_in), // active_set_up
									SR::with_len(
											veg::Tag<bool>{}, n_in), // new_active_constraints
									(do_ldlt && n_in > 0)
											? PROX_QP_ANY_OF({
														sparse_ldlt::add_row_req(
																xtag, itag, n_tot, false, n, n_tot),
														sparse_ldlt::delete_row_req(
																xtag, itag, n_tot, n_tot),
												})
											: refactorize_req,
							}),
							PROX_QP_ALL_OF({
									x_vec(n),    // Hdx
									x_vec(n_eq), // Adx
									x_vec(n_in), // Cdx
									x_vec(n),    // ATdy
									x_vec(n),    // CTdz
							}),
					}),
					line_search_req,
			});

			auto iter_req = PROX_QP_ANY_OF({
					PROX_QP_ALL_OF(
							{x_vec(n_eq), // primal_residual_eq_scaled
			         x_vec(n_in), // primal_residual_in_scaled_lo
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n_in), // primal_residual_in_scaled_up
			         x_vec(n),    // dual_residual_scaled
			         PROX_QP_ANY_OF({
									 unscaled_primal_dual_residual_req,
									 PROX_QP_ALL_OF({
											 x_vec(n),    // x_prev
											 x_vec(n_eq), // y_prev
											 x_vec(n_in), // z_prev
											 primal_dual_newton_semi_smooth_req,
									 }),
							 })}),
					refactorize_req, // mu_update
			});

			auto req = //
					PROX_QP_ALL_OF({
							x_vec(n),                             // g_scaled
							x_vec(n_eq),                          // b_scaled
							x_vec(n_in),                          // l_scaled
							x_vec(n_in),                          // u_scaled
							SR::with_len(veg::Tag<bool>{}, n_in), // active constr
							SR::with_len(itag, n_tot),            // kkt nnz counts
							refactorize_req,
							PROX_QP_ANY_OF({
									precond_req,
									PROX_QP_ALL_OF({
											do_ldlt ? PROX_QP_ALL_OF({
																		SR::with_len(itag, n_tot), // perm
																		SR::with_len(itag, n_tot), // etree
																		SR::with_len(itag, n_tot), // ldl nnz counts
																		SR::with_len(itag, lnnz), // ldl row indices
																		SR::with_len(xtag, lnnz), // ldl values
																})
															: PROX_QP_ALL_OF({
																		SR::with_len(itag, 0),
																		SR::with_len(xtag, 0),
																}),
											iter_req,
									}),
							}),
					});

			storage.resize_for_overwrite(req.alloc_req());
		}
	} _;

	QpWorkspace() = default;

	auto ldl_col_ptrs() const -> veg::Slice<I> {
		return _.ldl_col_ptrs.as_ref();
	}
	auto ldl_col_ptrs_mut() -> veg::SliceMut<I> {
		return _.ldl_col_ptrs.as_mut();
	}
	auto stack_mut() -> veg::dynstack::DynStackMut {
		return _.stack_mut();
	}

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

namespace detail {
template <typename T>
auto positive_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() > 0).select(expr, T::Zero(expr.rows())));
template <typename T>
auto negative_part(T const& expr)
		VEG_DEDUCE_RET((expr.array() < 0).select(expr, T::Zero(expr.rows())));

template <typename T, typename I>
VEG_NO_INLINE void noalias_gevmmv_add_impl( //
		ldlt::VectorViewMut<T> out_l,
		ldlt::VectorViewMut<T> out_r,
		sparse_ldlt::MatRef<T, I> a,
		ldlt::VectorView<T> in_l,
		ldlt::VectorView<T> in_r) {
	VEG_ASSERT_ALL_OF /* NOLINT */ (
			a.nrows() == out_r.dim,
			a.ncols() == in_r.dim,
			a.ncols() == out_l.dim,
			a.nrows() == in_l.dim);
	// equivalent to
	// out_r.to_eigen().noalias() += a.to_eigen() * in_r.to_eigen();
	// out_l.to_eigen().noalias() += a.to_eigen().transpose() * in_l.to_eigen();

	auto* ai = a.row_indices().ptr();
	auto* ax = a.values().ptr();
	auto n = a.ncols();

	for (usize j = 0; j < usize(n); ++j) {
		usize col_start = a.col_start(j);
		usize col_end = a.col_end(j);

		T acc0 = 0;
		T acc1 = 0;
		T acc2 = 0;
		T acc3 = 0;

		T in_rj = in_r(isize(j));

		usize pcount = col_end - col_start;

		usize p = col_start;

		auto zx = sparse_ldlt::util::zero_extend;

		for (; p < col_start + pcount / 4 * 4; p += 4) {
			auto i0 = isize(zx(ai[p + 0]));
			auto i1 = isize(zx(ai[p + 1]));
			auto i2 = isize(zx(ai[p + 2]));
			auto i3 = isize(zx(ai[p + 3]));

			T ai0j = ax[p + 0];
			T ai1j = ax[p + 1];
			T ai2j = ax[p + 2];
			T ai3j = ax[p + 3];

			out_r(i0) += ai0j * in_rj;
			out_r(i1) += ai1j * in_rj;
			out_r(i2) += ai2j * in_rj;
			out_r(i3) += ai3j * in_rj;

			acc0 += ai0j * in_l(i0);
			acc1 += ai1j * in_l(i1);
			acc2 += ai2j * in_l(i2);
			acc3 += ai3j * in_l(i3);
		}

		for (; p < col_end; ++p) {
			auto i = isize(zx(ai[p]));

			T aij = ax[p];
			out_r(i) += aij * in_rj;
			acc0 += aij * in_l(i);
		}

		acc0 = ((acc0 + acc1) + (acc2 + acc3));
		out_l(isize(j)) += acc0;
	}
}

template <typename T, typename I>
VEG_NO_INLINE void noalias_symhiv_add_impl( //
		ldlt::VectorViewMut<T> out,
		sparse_ldlt::MatRef<T, I> a,
		ldlt::VectorView<T> in) {
	VEG_ASSERT_ALL_OF /* NOLINT */ ( //
			a.nrows() == a.ncols(),
			a.nrows() == out.dim,
			a.ncols() == in.dim);
	// equivalent to
	// out.to_eigen().noalias() +=
	// 		a.to_eigen().template selfadjointView<Eigen::Upper>() * in.to_eigen();

	auto* ai = a.row_indices().ptr();
	auto* ax = a.values().ptr();
	auto n = a.ncols();

	for (usize j = 0; j < usize(n); ++j) {
		usize col_start = a.col_start(j);
		usize col_end = a.col_end(j);

		if (col_start == col_end) {
			continue;
		}

		T acc0 = 0;
		T acc1 = 0;
		T acc2 = 0;
		T acc3 = 0;

		T in_j = in(isize(j));

		usize pcount = col_end - col_start;

		auto zx = sparse_ldlt::util::zero_extend;

		if (zx(ai[col_end - 1]) == j) {
			T ajj = ax[col_end - 1];
			out(isize(j)) += ajj * in_j;
			pcount -= 1;
		}

		usize p = col_start;

		for (; p < col_start + pcount / 4 * 4; p += 4) {
			auto i0 = isize(zx(ai[p + 0]));
			auto i1 = isize(zx(ai[p + 1]));
			auto i2 = isize(zx(ai[p + 2]));
			auto i3 = isize(zx(ai[p + 3]));

			T ai0j = ax[p + 0];
			T ai1j = ax[p + 1];
			T ai2j = ax[p + 2];
			T ai3j = ax[p + 3];

			out(i0) += ai0j * in_j;
			out(i1) += ai1j * in_j;
			out(i2) += ai2j * in_j;
			out(i3) += ai3j * in_j;

			acc0 += ai0j * in(i0);
			acc1 += ai1j * in(i1);
			acc2 += ai2j * in(i2);
			acc3 += ai3j * in(i3);
		}
		for (; p < col_start + pcount; ++p) {
			auto i = isize(zx(ai[p]));

			T aij = ax[p];
			out(i) += aij * in_j;
			acc0 += aij * in(i);
		}
		acc0 = ((acc0 + acc1) + (acc2 + acc3));
		out(isize(j)) += acc0;
	}
}

/// noalias general vector matrix matrix vector add
template <typename OutL, typename OutR, typename A, typename InL, typename InR>
void noalias_gevmmv_add(
		OutL&& out_l, OutR&& out_r, A const& a, InL const& in_l, InR const& in_r) {
	detail::noalias_gevmmv_add_impl<typename A::Scalar, typename A::StorageIndex>(
			{ldlt::from_eigen, out_l},
			{ldlt::from_eigen, out_r},
			{sparse_ldlt::from_eigen, a},
			{ldlt::from_eigen, in_l},
			{ldlt::from_eigen, in_r});
}

/// noalias symmetric (hi) matrix vector add
template <typename Out, typename A, typename In>
void noalias_symhiv_add(Out&& out, A const& a, In const& in) {
	detail::noalias_symhiv_add_impl<typename A::Scalar, typename A::StorageIndex>(
			{ldlt::from_eigen, out},
			{sparse_ldlt::from_eigen, a},
			{ldlt::from_eigen, in});
}

template <typename T, typename I>
struct AugmentedKkt : Eigen::EigenBase<AugmentedKkt<T, I>> {
	struct Raw /* NOLINT */ {
		sparse_ldlt::MatRef<T, I> kkt_active;
		veg::Slice<bool> active_constraints;
		isize n;
		isize n_eq;
		isize n_in;
		T rho;
		T mu_eq;
		T mu_in;
	} _;

	AugmentedKkt /* NOLINT */ (Raw raw) noexcept : _{raw} {}

	using Scalar = T;
	using RealScalar = T;
	using StorageIndex = I;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false,
	};

	auto rows() const noexcept -> isize { return _.n + _.n_eq + _.n_in; }
	auto cols() const noexcept -> isize { return rows(); }
	template <typename Rhs>
	auto operator*(Eigen::MatrixBase<Rhs> const& x) const
			-> Eigen::Product<AugmentedKkt, Rhs, Eigen::AliasFreeProduct> {
		return Eigen::Product< //
				AugmentedKkt,
				Rhs,
				Eigen::AliasFreeProduct>{
				*this,
				x.derived(),
		};
	}
};

template <typename T>
using VecMapMut = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1>,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;
template <typename T>
using VecMap = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1> const,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;

template <typename V>
auto vec(V const& v) -> VecMap<typename V::Scalar> {
	static_assert(V::InnerStrideAtCompileTime == 1, ".");
	return {
			v.data(),
			v.rows(),
			v.cols(),
			Eigen::Stride<Eigen::Dynamic, 1>{
					v.outerStride(),
					v.innerStride(),
			},
	};
}

template <typename V>
auto vec_mut(V&& v) -> VecMapMut<typename veg::uncvref_t<V>::Scalar> {
	static_assert(veg::uncvref_t<V>::InnerStrideAtCompileTime == 1, ".");
	return {
			v.data(),
			v.rows(),
			v.cols(),
			Eigen::Stride<Eigen::Dynamic, 1>{
					v.outerStride(),
					v.innerStride(),
			},
	};
}

template <typename T, typename I>
auto middle_cols(
		sparse_ldlt::MatRef<T, I> mat, isize start, isize ncols, isize nnz)
		-> sparse_ldlt::MatRef<T, I> {
	VEG_ASSERT(start < mat.ncols());
	VEG_ASSERT(ncols <= mat.ncols() - start);
	return {
			sparse_ldlt::from_raw_parts,
			mat.nrows(),
			ncols,
			nnz,
			veg::Slice<I>{
					veg::unsafe,
					veg::from_raw_parts,
					mat.col_ptrs().ptr() + start,
					ncols + 1,
			},
			mat.row_indices(),
			mat.is_compressed() ? veg::Slice<I>{} : veg::Slice<I>{
					veg::unsafe,
					veg::from_raw_parts,
					mat.nnz_per_col().ptr() + start,
					ncols,
			},
			mat.values(),
	};
}

template <typename T, typename I>
auto middle_cols_mut(
		sparse_ldlt::MatMut<T, I> mat, isize start, isize ncols, isize nnz)
		-> sparse_ldlt::MatMut<T, I> {
	VEG_ASSERT(start < mat.ncols());
	VEG_ASSERT(ncols <= mat.ncols() - start);
	return {
			sparse_ldlt::from_raw_parts,
			mat.nrows(),
			ncols,
			nnz,
			veg::SliceMut<I>{
					veg::unsafe,
					veg::from_raw_parts,
					mat.col_ptrs_mut().ptr_mut() + start,
					ncols + 1,
			},
			mat.row_indices_mut(),
			mat.is_compressed() ? veg::SliceMut<I>{} : veg::SliceMut<I>{
					veg::unsafe,
					veg::from_raw_parts,
					mat.nnz_per_col_mut().ptr_mut() + start,
					ncols,
			},
			mat.values_mut(),
	};
}

template <typename T, typename I>
auto top_rows_unchecked(
		veg::Unsafe /*unsafe*/, sparse_ldlt::MatRef<T, I> mat, isize nrows)
		-> sparse_ldlt::MatRef<T, I> {
	VEG_ASSERT(nrows <= mat.nrows());
	return {
			sparse_ldlt::from_raw_parts,
			nrows,
			mat.ncols(),
			mat.nnz(),
			mat.col_ptrs(),
			mat.row_indices(),
			mat.nnz_per_col(),
			mat.values(),
	};
}

template <typename T, typename I>
auto top_rows_mut_unchecked(
		veg::Unsafe /*unsafe*/, sparse_ldlt::MatMut<T, I> mat, isize nrows)
		-> sparse_ldlt::MatMut<T, I> {
	VEG_ASSERT(nrows <= mat.nrows());
	return {
			sparse_ldlt::from_raw_parts,
			nrows,
			mat.ncols(),
			mat.nnz(),
			mat.col_ptrs_mut(),
			mat.row_indices_mut(),
			mat.nnz_per_col_mut(),
			mat.values_mut(),
	};
}

template <typename T, typename I>
auto ct_active(
		isize n, isize n_eq, isize n_in, sparse_ldlt::MatRef<T, I> kkt_active)
		-> sparse_ldlt::MatRef<T, I> {
	if (kkt_active.is_compressed()) {
	} else {
		return {
				sparse_ldlt::from_raw_parts,
				n,
				n_in,
				0, // nnz not used
				{
						veg::unsafe,
						veg::from_raw_parts,
						kkt_active.col_ptrs().ptr() + n + n_eq,
						n_in + 1,
				},
				kkt_active.row_indices(),
				{
						veg::unsafe,
						veg::from_raw_parts,
						kkt_active.nnz_per_col().ptr() + n + n_eq,
						n_in,
				},
				kkt_active.values(),
		};
	}
}

template <typename T, typename I, typename P>
auto unscaled_primal_dual_residual(
		VecMapMut<T> primal_residual_eq_scaled,
		VecMapMut<T> primal_residual_in_scaled_lo,
		VecMapMut<T> primal_residual_in_scaled_up,
		VecMapMut<T> dual_residual_scaled,
		T& primal_feasibility_eq_rhs_0,
		T& primal_feasibility_in_rhs_0,
		T dual_feasibility_rhs_0,
		T dual_feasibility_rhs_1,
		T dual_feasibility_rhs_3,
		P& precond,
		QpView<T, I> qp,
		sparse_ldlt::MatRef<T, I> H_scaled,
		sparse_ldlt::MatRef<T, I> AT_scaled,
		sparse_ldlt::MatRef<T, I> CT_scaled,
		VecMap<T> g_scaled_e,
		VecMap<T> x_e,
		VecMap<T> y_e,
		VecMap<T> z_e,
		veg::dynstack::DynStackMut stack) -> veg::Tuple<T, T> {
	isize n = x_e.rows();

	LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
	dual_residual_scaled = g_scaled_e;

	{
		tmp.setZero();
		detail::noalias_symhiv_add(tmp, H_scaled.to_eigen(), x_e);
		dual_residual_scaled += tmp;

		precond.unscale_dual_residual_in_place({ldlt::from_eigen, tmp});
		dual_feasibility_rhs_0 = infty_norm(tmp);
	}

	{
		auto ATy = tmp;
		ATy.setZero();
		primal_residual_eq_scaled.setZero();

		detail::noalias_gevmmv_add(
				primal_residual_eq_scaled, ATy, AT_scaled.to_eigen(), x_e, y_e);

		dual_residual_scaled += ATy;

		precond.unscale_dual_residual_in_place({ldlt::from_eigen, ATy});
		dual_feasibility_rhs_1 = infty_norm(ATy);
	}

	{
		auto CTz = tmp;
		CTz.setZero();
		primal_residual_in_scaled_up.setZero();

		detail::noalias_gevmmv_add(
				primal_residual_in_scaled_up, CTz, CT_scaled.to_eigen(), x_e, z_e);

		dual_residual_scaled += CTz;

		precond.unscale_dual_residual_in_place({ldlt::from_eigen, CTz});
		dual_feasibility_rhs_3 = infty_norm(CTz);
	}

	precond.unscale_primal_residual_in_place_eq(
			{ldlt::from_eigen, primal_residual_eq_scaled});

	primal_feasibility_eq_rhs_0 = infty_norm(primal_residual_eq_scaled);

	precond.unscale_primal_residual_in_place_in(
			{ldlt::from_eigen, primal_residual_in_scaled_up});
	primal_feasibility_in_rhs_0 = infty_norm(primal_residual_in_scaled_up);

	auto b = qp.b.to_eigen();
	auto l = qp.l.to_eigen();
	auto u = qp.u.to_eigen();
	primal_residual_in_scaled_lo =
			detail::positive_part(primal_residual_in_scaled_up - u) +
			detail::negative_part(primal_residual_in_scaled_up - l);

	primal_residual_eq_scaled -= b;
	T primal_feasibility_eq_lhs = infty_norm(primal_residual_eq_scaled);
	T primal_feasibility_in_lhs = infty_norm(primal_residual_in_scaled_lo);
	T primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	// scaled Ax - b
	precond.scale_primal_residual_in_place_eq(
			{ldlt::from_eigen, primal_residual_eq_scaled});
	// scaled Cx
	precond.scale_primal_residual_in_place_in(
			{ldlt::from_eigen, primal_residual_in_scaled_up});

	precond.unscale_dual_residual_in_place(
			{ldlt::from_eigen, dual_residual_scaled});
	T dual_feasibility_lhs = infty_norm(dual_residual_scaled);
	precond.scale_dual_residual_in_place(
			{ldlt::from_eigen, dual_residual_scaled});

	return veg::tuplify(primal_feasibility_lhs, dual_feasibility_lhs);
}
} // namespace detail

template <typename T, typename I, typename P>
void qp_setup(QpWorkspace<T, I>& work, QpView<T, I> qp, P& /*precond*/) {
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
		Settings<T> const& settings,
		P& precond,
		QpView<T, I> qp) {

	using namespace veg::literals;
	namespace util = sparse_ldlt::util;
	auto zx = util::zero_extend;

	veg::dynstack::DynStackMut stack = work.stack_mut();

	isize n = qp.H.nrows();
	isize n_eq = qp.AT.ncols();
	isize n_in = qp.CT.ncols();
	isize n_tot = n + n_eq + n_in;

	sparse_ldlt::MatMut<T, I> kkt = work.kkt_mut();

	auto kkt_top_n_rows = detail::top_rows_mut_unchecked(veg::unsafe, kkt, n);

	sparse_ldlt::MatMut<T, I> H_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, 0, n, qp.H.nnz());

	sparse_ldlt::MatMut<T, I> AT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n, n_eq, qp.AT.nnz());

	sparse_ldlt::MatMut<T, I> CT_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, n + n_eq, n_in, qp.CT.nnz());

	auto H_scaled_e = H_scaled.to_eigen();
	LDLT_TEMP_VEC_UNINIT(T, g_scaled_e, n, stack);
	auto A_scaled_e = AT_scaled.to_eigen().transpose();
	LDLT_TEMP_VEC_UNINIT(T, b_scaled_e, n_eq, stack);
	auto C_scaled_e = CT_scaled.to_eigen().transpose();
	LDLT_TEMP_VEC_UNINIT(T, l_scaled_e, n_in, stack);
	LDLT_TEMP_VEC_UNINIT(T, u_scaled_e, n_in, stack);

	g_scaled_e = qp.g.to_eigen();
	b_scaled_e = qp.b.to_eigen();
	l_scaled_e = qp.l.to_eigen();
	u_scaled_e = qp.u.to_eigen();

	QpViewMut<T, I> qp_scaled = {
			H_scaled,
			{sparse_ldlt::from_eigen, g_scaled_e},
			AT_scaled,
			{sparse_ldlt::from_eigen, b_scaled_e},
			CT_scaled,
			{sparse_ldlt::from_eigen, l_scaled_e},
			{sparse_ldlt::from_eigen, u_scaled_e},
	};

	precond.scale_qp_in_place(qp_scaled, stack);

	T const primal_feasibility_rhs_1_eq = infty_norm(qp.b.to_eigen());
	T const primal_feasibility_rhs_1_in_u = infty_norm(qp.u.to_eigen());
	T const primal_feasibility_rhs_1_in_l = infty_norm(qp.l.to_eigen());
	T const dual_feasibility_rhs_2 = infty_norm(qp.g.to_eigen());

	auto ldl_col_ptrs = work.ldl_col_ptrs_mut();
	auto max_lnnz = isize(zx(ldl_col_ptrs[n_tot]));

	veg::Tag<I> itag;
	veg::Tag<T> xtag;

	auto _active_constraints = stack.make_new(veg::Tag<bool>{}, n_in).unwrap();
	auto _kkt_nnz_counts = stack.make_new_for_overwrite(itag, n_tot).unwrap();

	bool do_ldlt = work._.do_ldlt;

	isize ldlt_ntot = do_ldlt ? n_tot : 0;
	isize ldlt_lnnz = do_ldlt ? max_lnnz : 0;

	auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _etree = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _ldl_nnz_counts = stack.make_new_for_overwrite(itag, ldlt_ntot).unwrap();
	auto _ldl_row_indices =
			stack.make_new_for_overwrite(itag, ldlt_lnnz).unwrap();
	auto _ldl_values = stack.make_new_for_overwrite(xtag, ldlt_lnnz).unwrap();

	veg::Slice<I> perm_inv = work._.perm_inv.as_ref();
	veg::SliceMut<I> perm = _perm.as_mut();

	if (do_ldlt) {
		// compute perm from perm_inv
		for (isize i = 0; i < n_tot; ++i) {
			perm[isize(zx(perm_inv[i]))] = I(i);
		}
	}

	veg::SliceMut<I> kkt_nnz_counts = _kkt_nnz_counts.as_mut();

	// H and A are always active
	for (usize j = 0; j < usize(n + n_eq); ++j) {
		kkt_nnz_counts[isize(j)] = I(kkt.col_end(j) - kkt.col_start(j));
	}
	// ineq constraints initially inactive
	for (isize j = 0; j < n_in; ++j) {
		kkt_nnz_counts[n + n_eq + j] = 0;
	}

	Eigen::MINRES<
			detail::AugmentedKkt<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>
			iterative_solver;

	sparse_ldlt::MatMut<T, I> kkt_active = {
			sparse_ldlt::from_raw_parts,
			n_tot,
			n_tot,
			qp.H.nnz() + qp.AT.nnz(),
			kkt.col_ptrs_mut(),
			kkt.row_indices_mut(),
			kkt_nnz_counts,
			kkt.values_mut(),
	};

	veg::SliceMut<I> etree = _etree.as_mut();
	veg::SliceMut<I> ldl_nnz_counts = _ldl_nnz_counts.as_mut();
	veg::SliceMut<I> ldl_row_indices = _ldl_row_indices.as_mut();
	veg::SliceMut<T> ldl_values = _ldl_values.as_mut();
	veg::SliceMut<bool> active_constraints = _active_constraints.as_mut();

	sparse_ldlt::MatMut<T, I> ldl = {
			sparse_ldlt::from_raw_parts,
			n_tot,
			n_tot,
			0,
			ldl_col_ptrs,
			ldl_row_indices,
			do_ldlt ? ldl_nnz_counts : veg::SliceMut<I>{},
			ldl_values,
	};

	T rho = T(1e-6);
	T mu_eq = T(1e3);
	T mu_in = T(1e1);

	detail::AugmentedKkt<T, I> aug_kkt{{
			kkt_active.as_const(),
			active_constraints.as_const(),
			n,
			n_eq,
			n_in,
			rho,
			mu_eq,
			mu_in,
	}};

	T bcl_eta_ext_init = pow(T(0.1), settings.alpha_bcl);
	T bcl_eta_ext = bcl_eta_ext_init;
	T bcl_eta_in(1);
	T eps_in_min = std::min(settings.eps_abs, T(1e-9));

	using DMat = Eigen::Matrix<T, -1, -1>;
	auto inner_reconstructed_matrix = [&]() -> DMat {
		VEG_ASSERT(do_ldlt);
		auto ldl_dense = ldl.to_eigen().toDense();
		auto l = DMat(ldl_dense.template triangularView<Eigen::UnitLower>());
		auto lt = l.transpose();
		auto d = ldl_dense.diagonal().asDiagonal();
		auto mat = DMat(l * d * lt);
		return mat;
	};
	auto reconstructed_matrix = [&]() -> DMat {
		auto mat = inner_reconstructed_matrix();
		auto mat_backup = mat;
		for (isize i = 0; i < n_tot; ++i) {
			for (isize j = 0; j < n_tot; ++j) {
				mat(i, j) = mat_backup(perm_inv[i], perm_inv[j]);
			}
		}
		return mat;
	};

	auto reconstruction_error = [&]() -> DMat {
		auto diff = DMat(
				reconstructed_matrix() -
				DMat(DMat(kkt_active.to_eigen())
		             .template selfadjointView<Eigen::Upper>()));
		diff.diagonal().head(n).array() -= rho;
		diff.diagonal().segment(n, n_eq).array() -= -1 / mu_eq;
		for (isize i = 0; i < n_in; ++i) {
			diff.diagonal()[n + n_eq + i] -=
					active_constraints[i] ? -1 / mu_in : T(1);
		}
		return diff;
	};
	veg::unused(reconstruction_error);

	auto refactorize = [&]() -> void {
		if (do_ldlt) {
			sparse_ldlt::factorize_symbolic_non_zeros(
					ldl_nnz_counts,
					etree,
					work._.perm_inv.as_mut(),
					perm.as_const(),
					kkt_active.symbolic(),
					stack);

			auto _diag = stack.make_new_for_overwrite(xtag, n_tot).unwrap();
			T* diag = _diag.ptr_mut();

			for (isize i = 0; i < n; ++i) {
				diag[i] = rho;
			}
			for (isize i = 0; i < n_eq; ++i) {
				diag[n + i] = -1 / mu_eq;
			}
			for (isize i = 0; i < n_in; ++i) {
				diag[(n + n_eq) + i] = active_constraints[i] ? -1 / mu_in : T(1);
			}

			sparse_ldlt::factorize_numeric(
					ldl_values.ptr_mut(),
					ldl_row_indices.ptr_mut(),
					diag,
					perm.ptr(),
					ldl_col_ptrs.as_const(),
					etree.as_const(),
					perm_inv,
					kkt_active.as_const(),
					stack);
			isize ldl_nnz = 0;
			for (isize i = 0; i < n_tot; ++i) {
				ldl_nnz =
						util::checked_non_negative_plus(ldl_nnz, isize(ldl_nnz_counts[i]));
			}
			ldl._set_nnz(ldl_nnz);
		} else {
			aug_kkt = {{
					kkt_active.as_const(),
					active_constraints.as_const(),
					n,
					n_eq,
					n_in,
					rho,
					mu_eq,
					mu_in,
			}};
			iterative_solver.compute(aug_kkt);
		}
	};
	refactorize();

	auto x_e = x.to_eigen();
	auto y_e = y.to_eigen();
	auto z_e = z.to_eigen();

	auto ldl_solve = [&](VectorViewMut<T> sol, VectorView<T> rhs) -> void {
		LDLT_TEMP_VEC_UNINIT(T, work, n_tot, stack);
		auto rhs_e = rhs.to_eigen();
		auto sol_e = sol.to_eigen();

		if (do_ldlt) {

			for (isize i = 0; i < n_tot; ++i) {
				work[i] = rhs_e[isize(zx(perm[i]))];
			}

			sparse_ldlt::dense_lsolve<T, I>( //
					{sparse_ldlt::from_eigen, work},
					ldl.as_const());

			for (isize i = 0; i < n_tot; ++i) {
				work[i] /= ldl_values[isize(zx(ldl_col_ptrs[i]))];
			}

			sparse_ldlt::dense_ltsolve<T, I>( //
					{sparse_ldlt::from_eigen, work},
					ldl.as_const());

			for (isize i = 0; i < n_tot; ++i) {
				sol_e[i] = work[isize(zx(perm_inv[i]))];
			}
		} else {
			work = iterative_solver.solve(rhs_e);
			sol_e = work;
		}
	};

	auto ldl_iter_solve_noalias = [&](VectorViewMut<T> sol,
	                                  VectorView<T> rhs,
	                                  VectorView<T> init_guess) -> void {
		auto rhs_e = rhs.to_eigen();
		auto sol_e = sol.to_eigen();

		if (init_guess.dim == sol.dim) {
			sol_e = init_guess.to_eigen();
		} else {
			sol_e.setZero();
		}

		LDLT_TEMP_VEC_UNINIT(T, err, n_tot, stack);

		T prev_err_norm = std::numeric_limits<T>::infinity();

		for (isize solve_iter = 0; solve_iter < settings.nb_iterative_refinement;
		     ++solve_iter) {

			auto err_x = err.head(n);
			auto err_y = err.segment(n, n_eq);
			auto err_z = err.tail(n_in);

			auto sol_x = sol_e.head(n);
			auto sol_y = sol_e.segment(n, n_eq);
			auto sol_z = sol_e.tail(n_in);

			err = -rhs_e;

			if (solve_iter > 0) {
				detail::noalias_symhiv_add(err, kkt_active.to_eigen(), sol_e);
				err_x += rho * sol_x;
				err_y += (-1 / mu_eq) * sol_y;
				for (isize i = 0; i < n_in; ++i) {
					err_z[i] += (active_constraints[i] ? -1 / mu_in : T(1)) * sol_z[i];
				}
			}

			T err_norm = infty_norm(err);
			if (err_norm > prev_err_norm / T(2)) {
				break;
			}
			prev_err_norm = err_norm;

			ldl_solve({ldlt::from_eigen, err}, {ldlt::from_eigen, err});

			sol_e -= err;
		}
	};

	auto ldl_solve_in_place = [&](VectorViewMut<T> rhs,
	                              VectorView<T> init_guess) {
		LDLT_TEMP_VEC_UNINIT(T, tmp, n_tot, stack);
		ldl_iter_solve_noalias({ldlt::from_eigen, tmp}, rhs.as_const(), init_guess);
		rhs.to_eigen() = tmp;
	};

	if (!settings.warm_start) {
		LDLT_TEMP_VEC_UNINIT(T, rhs, n_tot, stack);
		LDLT_TEMP_VEC_UNINIT(T, no_guess, 0, stack);

		rhs.head(n) = -g_scaled_e;
		rhs.segment(n, n_eq) = b_scaled_e;
		rhs.segment(n + n_eq, n_in).setZero();

		ldl_solve_in_place({ldlt::from_eigen, rhs}, {ldlt::from_eigen, no_guess});
		x_e = rhs.head(n);
		y_e = rhs.segment(n, n_eq);
		z_e = rhs.segment(n + n_eq, n_in);
	}

	for (isize iter = 0; iter < settings.max_iter; ++iter) {
		T new_bcl_mu_eq = mu_eq;
		T new_bcl_mu_in = mu_in;

		{
			T primal_feasibility_eq_rhs_0;
			T primal_feasibility_in_rhs_0;

			T dual_feasibility_rhs_0(0);
			T dual_feasibility_rhs_1(0);
			T dual_feasibility_rhs_3(0);

			LDLT_TEMP_VEC_UNINIT(T, primal_residual_eq_scaled, n_eq, stack);
			LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_lo, n_in, stack);
			LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled_up, n_in, stack);

			LDLT_TEMP_VEC_UNINIT(T, dual_residual_scaled, n, stack);

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto is_primal_feasible = [&](T primal_feasibility_lhs) -> bool {
				T rhs_pri = settings.eps_abs;
				if (settings.eps_rel != 0) {
					rhs_pri += settings.eps_rel * std::max({
																						primal_feasibility_eq_rhs_0,
																						primal_feasibility_in_rhs_0,
																						primal_feasibility_rhs_1_eq,
																						primal_feasibility_rhs_1_in_l,
																						primal_feasibility_rhs_1_in_u,
																				});
				}
				return primal_feasibility_lhs <= rhs_pri;
			};
			auto is_dual_feasible = [&](T dual_feasibility_lhs) -> bool {
				T rhs_dua = settings.eps_abs;
				if (settings.eps_rel != 0) {
					rhs_dua += settings.eps_rel * std::max({
																						dual_feasibility_rhs_0,
																						dual_feasibility_rhs_1,
																						dual_feasibility_rhs_2,
																						dual_feasibility_rhs_3,
																				});
				}

				return dual_feasibility_lhs <= rhs_dua;
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			VEG_BIND(
					auto,
					(primal_feasibility_lhs, dual_feasibility_lhs),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							qp,
							H_scaled.as_const(),
							AT_scaled.as_const(),
							CT_scaled.as_const(),
							detail::vec(g_scaled_e),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));

			if (settings.verbose) {
				std::cout << "-------- outer iteration: " << iter << " primal residual "
									<< primal_feasibility_lhs << " dual residual "
									<< dual_feasibility_lhs << " mu_in " << mu_in
									<< " bcl_eta_ext " << bcl_eta_ext << " bcl_eta_in "
									<< bcl_eta_in << std::endl;
			}
			if (is_primal_feasible(primal_feasibility_lhs) &&
			    is_dual_feasible(dual_feasibility_lhs)) {
				break;
			}

			LDLT_TEMP_VEC_UNINIT(T, x_prev_e, n, stack);
			LDLT_TEMP_VEC_UNINIT(T, y_prev_e, n_eq, stack);
			LDLT_TEMP_VEC_UNINIT(T, z_prev_e, n_in, stack);
			LDLT_TEMP_VEC(T, dw_prev, n_tot, stack);

			x_prev_e = x_e;
			y_prev_e = y_e;
			z_prev_e = z_e;

			// Cx + 1/mu_in * z_prev
			primal_residual_in_scaled_up += 1 / mu_in * z_prev_e;
			primal_residual_in_scaled_lo = primal_residual_in_scaled_up;

			// Cx - l + 1/mu_in * z_prev
			primal_residual_in_scaled_lo -= l_scaled_e;

			// Cx - u + 1/mu_in * z_prev
			primal_residual_in_scaled_up -= u_scaled_e;

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto primal_dual_newton_semi_smooth = [&]() -> void {
				for (isize iter_inner = 0; iter_inner < settings.max_iter_in;
				     ++iter_inner) {
					LDLT_TEMP_VEC_UNINIT(T, dw, n_tot, stack);

					if (settings.verbose) {
						std::cout
								<< "-------------------starting inner loop solve in place "
								<< std::endl;
					}
					// primal_dual_semi_smooth_newton_step
					{
						LDLT_TEMP_VEC_UNINIT(bool, active_set_lo, n_in, stack);
						LDLT_TEMP_VEC_UNINIT(bool, active_set_up, n_in, stack);
						LDLT_TEMP_VEC_UNINIT(bool, new_active_constraints, n_in, stack);
						auto rhs = dw;

						active_set_lo.array() = primal_residual_in_scaled_lo.array() <= 0;
						active_set_up.array() = primal_residual_in_scaled_up.array() >= 0;
						new_active_constraints = active_set_lo || active_set_up;

						// active set change
						if (n_in > 0) {
							bool removed = false;
							bool added = false;
							veg::unused(removed, added);

							for (isize i = 0; i < n_in; ++i) {
								bool was_active = active_constraints[i];
								bool is_active = new_active_constraints[i];

								isize idx = n + n_eq + i;

								usize col_nnz =
										zx(kkt.col_end(usize(idx))) - zx(kkt.col_start(usize(idx)));

								if (is_active && !was_active) {
									added = true;

									kkt_active.nnz_per_col_mut()[idx] = I(col_nnz);
									kkt_active._set_nnz(kkt_active.nnz() + isize(col_nnz));

									if (do_ldlt) {
										sparse_ldlt::VecRef<T, I> new_col{
												sparse_ldlt::from_raw_parts,
												n_tot,
												{
														veg::unsafe,
														veg::from_raw_parts,
														kkt.row_indices().ptr() +
																zx(kkt.col_start(usize(idx))),
														isize(col_nnz),
												},
												{
														veg::unsafe,
														veg::from_raw_parts,
														kkt.values().ptr() + zx(kkt.col_start(usize(idx))),
														isize(col_nnz),
												},
										};

										ldl = sparse_ldlt::add_row(
												ldl, etree, perm_inv, idx, new_col, -1 / mu_in, stack);
									}
									active_constraints[i] = new_active_constraints[i];

								} else if (!is_active && was_active) {
									removed = true;
									kkt_active.nnz_per_col_mut()[idx] = 0;
									kkt_active._set_nnz(kkt_active.nnz() - isize(col_nnz));
									if (do_ldlt) {
										ldl = sparse_ldlt::delete_row(
												ldl, etree, perm_inv, idx, stack);
									}
									active_constraints[i] = new_active_constraints[i];
								}
							}

							if (!do_ldlt) {
								if (removed || added) {
									refactorize();
								}
							}
						}

						rhs.head(n) = -dual_residual_scaled;
						rhs.segment(n, n_eq) = -primal_residual_eq_scaled;

						for (isize i = 0; i < n_in; ++i) {
							if (active_set_up(i)) {
								rhs(n + n_eq + i) =
										1 / mu_in * z_e(i) - primal_residual_in_scaled_up(i);
							} else if (active_set_lo(i)) {
								rhs(n + n_eq + i) =
										1 / mu_in * z_e(i) - primal_residual_in_scaled_lo(i);
							} else {
								rhs(n + n_eq + i) = -z_e(i);
								rhs.head(n) += z_e(i) * C_scaled_e.row(i);
							}
						}

						ldl_solve_in_place(
								{ldlt::from_eigen, rhs}, {ldlt::from_eigen, dw_prev});
					}
					if (settings.verbose) {
						std::cout
								<< "-------------------finished inner loop solve in place "
								<< std::endl;
					}
					auto dx = dw.head(n);
					auto dy = dw.segment(n, n_eq);
					auto dz = dw.segment(n + n_eq, n_in);

					LDLT_TEMP_VEC(T, Hdx, n, stack);
					LDLT_TEMP_VEC(T, Adx, n_eq, stack);
					LDLT_TEMP_VEC(T, Cdx, n_in, stack);

					LDLT_TEMP_VEC(T, ATdy, n, stack);
					LDLT_TEMP_VEC(T, CTdz, n, stack);

					detail::noalias_symhiv_add(Hdx, H_scaled_e, dx);
					Adx.noalias() += A_scaled_e * dx;
					ATdy.noalias() += A_scaled_e.transpose() * dy;
					Cdx.noalias() += C_scaled_e * dx;
					CTdz.noalias() += C_scaled_e.transpose() * dz;

					T alpha = 1;
					// primal dual line search
					if (settings.verbose) {
						std::cout << "-------------------starting inner loop line search "
											<< std::endl;
					}
					if (n_in > 0) {
						auto primal_dual_gradient_norm =
								[&](T alpha_cur) -> PrimalDualGradResult<T> {
							LDLT_TEMP_VEC_UNINIT(T, Cdx_active, n_in, stack);
							LDLT_TEMP_VEC_UNINIT(T, active_part_z, n_in, stack);
							{
								LDLT_TEMP_VEC_UNINIT(T, tmp_lo, n_in, stack);
								LDLT_TEMP_VEC_UNINIT(T, tmp_up, n_in, stack);

								auto zero = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_in);

								tmp_lo = primal_residual_in_scaled_lo + alpha_cur * Cdx;
								tmp_up = primal_residual_in_scaled_up + alpha_cur * Cdx;
								Cdx_active = (tmp_lo.array() < 0 || tmp_up.array() > 0)
								                 .select(Cdx, zero);
								active_part_z =
										(tmp_lo.array() < 0)
												.select(primal_residual_in_scaled_lo, zero) +
										(tmp_up.array() > 0)
												.select(primal_residual_in_scaled_up, zero);
							}

							T nu = 1;

							T a = dx.dot(Hdx) +                                   //
							      rho * dx.squaredNorm() +                        //
							      mu_eq * Adx.squaredNorm() +                     //
							      mu_in * Cdx_active.squaredNorm() +              //
							      nu / mu_eq * (mu_eq * Adx - dy).squaredNorm() + //
							      nu / mu_in * (mu_in * Cdx_active - dz).squaredNorm();

							T b = x_e.dot(Hdx) +                                         //
							      (rho * (x_e - x_prev_e) + g_scaled_e).dot(dx) +        //
							      Adx.dot(mu_eq * primal_residual_eq_scaled + y_e) +     //
							      mu_in * Cdx_active.dot(active_part_z) +                //
							      nu * primal_residual_eq_scaled.dot(mu_eq * Adx - dy) + //
							      nu * (active_part_z - 1 / mu_in * z_e)
							               .dot(mu_in * Cdx_active - dz);

							return {
									a,
									b,
									a * alpha_cur + b,
							};
						};

						LDLT_TEMP_VEC_UNINIT(T, alphas, 2 * n_in, stack);
						isize alphas_count = 0;

						for (isize i = 0; i < n_in; ++i) {
							T alpha_candidates[2] = {
									-primal_residual_in_scaled_lo(i) / (Cdx(i)),
									-primal_residual_in_scaled_up(i) / (Cdx(i)),
							};

							for (auto alpha_candidate : alpha_candidates) {
								if (alpha_candidate > 0) {
									alphas[alphas_count] = alpha_candidate;
									++alphas_count;
								}
							}
						}
						std::sort(alphas.data(), alphas.data() + alphas_count);
						alphas_count =
								std::unique(alphas.data(), alphas.data() + alphas_count) -
								alphas.data();

						if (alphas_count > 0 && alphas[0] <= 1) {
							auto infty = std::numeric_limits<T>::infinity();

							T last_neg_grad = 0;
							T alpha_last_neg = 0;
							T first_pos_grad = 0;
							T alpha_first_pos = infty;

							{
								for (isize i = 0; i < alphas_count; ++i) {
									T alpha_cur = alphas[i];
									T gr = primal_dual_gradient_norm(alpha_cur).grad;

									if (gr < 0) {
										alpha_last_neg = alpha_cur;
										last_neg_grad = gr;
									} else {
										first_pos_grad = gr;
										alpha_first_pos = alpha_cur;
										break;
									}
								}

								if (alpha_last_neg == 0) {
									last_neg_grad =
											primal_dual_gradient_norm(alpha_last_neg).grad;
								}

								if (alpha_first_pos == infty) {
									auto res = primal_dual_gradient_norm(2 * alpha_last_neg + 1);
									alpha = -res.b / res.a;
								} else {
									alpha = alpha_last_neg -
									        last_neg_grad * (alpha_first_pos - alpha_last_neg) /
									            (first_pos_grad - last_neg_grad);
								}
							}
						} else {
							auto res = primal_dual_gradient_norm(T(0));
							alpha = -res.b / res.a;
						}
					}
					if (alpha * infty_norm(dw) < T(1e-11) && iter > 0) {
						return;
					}
					if (settings.verbose) {
						std::cout << "-------------------finished inner loop line search "
											<< std::endl;
					}

					x_e += alpha * dx;
					y_e += alpha * dy;
					z_e += alpha * dz;

					dual_residual_scaled += alpha * (Hdx + ATdy + CTdz + rho * dx);
					primal_residual_eq_scaled += alpha * (Adx - 1 / mu_eq * dy);
					primal_residual_in_scaled_lo += alpha * Cdx;
					primal_residual_in_scaled_up += alpha * Cdx;

					T err_in = std::max({
							(infty_norm(
									detail::negative_part(primal_residual_in_scaled_lo) +
									detail::positive_part(primal_residual_in_scaled_up) -
									1 / mu_in * z_e)),
							(infty_norm(primal_residual_eq_scaled)),
							(infty_norm(dual_residual_scaled)),
					});
					if (settings.verbose) {
						std::cout << "--inner iter " << iter_inner << " iner error "
											<< err_in << " alpha " << alpha << " infty_norm(dw) "
											<< infty_norm(dw) << std::endl;
					}
					if (err_in <= bcl_eta_in) {
						return;
					}
				}
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

			primal_dual_newton_semi_smooth();

			VEG_BIND(
					auto,
					(primal_feasibility_lhs_new, dual_feasibility_lhs_new),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							qp,
							H_scaled.as_const(),
							AT_scaled.as_const(),
							CT_scaled.as_const(),
							detail::vec(g_scaled_e),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));
			if (is_primal_feasible(primal_feasibility_lhs_new) &&
			    is_dual_feasible(dual_feasibility_lhs_new)) {
				break;
			}

			// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
			auto bcl_update = [&]() -> void {
				if (primal_feasibility_lhs_new <= bcl_eta_ext) {
					bcl_eta_ext *= 1 / pow(mu_in, settings.beta_bcl);
					bcl_eta_in = std::max(bcl_eta_in / mu_in, eps_in_min);
				} else {
					y_e = y_prev_e;
					z_e = z_prev_e;
					new_bcl_mu_in =
							std::min(mu_in * settings.mu_update_factor, settings.mu_max_in);
					new_bcl_mu_eq =
							std::min(mu_eq * settings.mu_update_factor, settings.mu_max_eq);
					bcl_eta_ext =
							bcl_eta_ext_init / pow(new_bcl_mu_in, settings.alpha_bcl);
					bcl_eta_in = 1 / std::max(new_bcl_mu_in, eps_in_min);
				}
			};
			// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
			bcl_update();

			VEG_BIND(
					auto,
					(_, dual_feasibility_lhs_new_2),
					detail::unscaled_primal_dual_residual(
							primal_residual_eq_scaled,
							primal_residual_in_scaled_lo,
							primal_residual_in_scaled_up,
							dual_residual_scaled,
							primal_feasibility_eq_rhs_0,
							primal_feasibility_in_rhs_0,
							dual_feasibility_rhs_0,
							dual_feasibility_rhs_1,
							dual_feasibility_rhs_3,
							precond,
							qp,
							H_scaled.as_const(),
							AT_scaled.as_const(),
							CT_scaled.as_const(),
							detail::vec(g_scaled_e),
							detail::vec(x_e),
							detail::vec(y_e),
							detail::vec(z_e),
							stack));
      veg::unused(_);

			if (primal_feasibility_lhs_new >= primal_feasibility_lhs && //
			    dual_feasibility_lhs_new_2 >= primal_feasibility_lhs && //
			    mu_in >= 1.E5) {
				new_bcl_mu_in = settings.cold_reset_mu_in;
				new_bcl_mu_eq = settings.cold_reset_mu_eq;
			}
		}

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		auto mu_update = [&]() -> void { refactorize(); };
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		if (mu_in != new_bcl_mu_in || mu_eq != new_bcl_mu_eq) {
			mu_eq = new_bcl_mu_eq;
			mu_in = new_bcl_mu_in;
			mu_update();
		}
	}

	precond.unscale_primal_in_place({ldlt::from_eigen, x_e});
	precond.unscale_dual_in_place_eq({ldlt::from_eigen, y_e});
	precond.unscale_dual_in_place_in({ldlt::from_eigen, z_e});
}
} // namespace sparse
} // namespace qp

namespace Eigen {
namespace internal {
template <typename T, typename I>
struct traits<qp::sparse::detail::AugmentedKkt<T, I>>
		: Eigen::internal::traits<Eigen::SparseMatrix<T, Eigen::ColMajor, I>> {};

template <typename Rhs, typename T, typename I>
struct generic_product_impl<
		qp::sparse::detail::AugmentedKkt<T, I>,
		Rhs,
		SparseShape,
		DenseShape,
		GemvProduct>
		: generic_product_impl_base<
					qp::sparse::detail::AugmentedKkt<T, I>,
					Rhs,
					generic_product_impl<qp::sparse::detail::AugmentedKkt<T, I>, Rhs>> {
	using Mat_ = qp::sparse::detail::AugmentedKkt<T, I>;

	using Scalar = typename Product<Mat_, Rhs>::Scalar;

	template <typename Dst>
	static void scaleAndAddTo(
			Dst& dst, Mat_ const& lhs, Rhs const& rhs, Scalar const& alpha) {
		using veg::isize;

		VEG_ASSERT(alpha == Scalar(1));
		qp::sparse::detail::noalias_symhiv_add(
				dst, lhs._.kkt_active.to_eigen(), rhs);

		{
			isize n = lhs._.n;
			isize n_eq = lhs._.n_eq;
			isize n_in = lhs._.n_in;

			auto dst_x = dst.head(n);
			auto dst_y = dst.segment(n, n_eq);
			auto dst_z = dst.tail(n_in);

			auto rhs_x = rhs.head(n);
			auto rhs_y = rhs.segment(n, n_eq);
			auto rhs_z = rhs.tail(n_in);

			dst_x += lhs._.rho * rhs_x;
			dst_y += (-1 / lhs._.mu_eq) * rhs_y;
			for (isize i = 0; i < n_in; ++i) {
				dst_z[i] +=
						(lhs._.active_constraints[i] ? -1 / lhs._.mu_in : T(1)) * rhs_z[i];
			}
		}
	}
};
} // namespace internal
} // namespace Eigen

#endif /* end of include guard INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS */
