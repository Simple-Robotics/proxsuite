#ifndef INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS
#define INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS

#include <sparse_ldlt/core.hpp>
#include <sparse_ldlt/factorize.hpp>
#include <sparse_ldlt/update.hpp>
#include <sparse_ldlt/rowmod.hpp>
#include <qp/views.hpp>

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
	sparse_ldlt::MatMut<T, T> H;
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

namespace preconditionner {
enum struct Symmetry {
	LOWER,
	UPPER,
};

namespace detail {
template <typename T, typename I>
void rowwise_infty_norm(T* row_norm, sparse_ldlt::MatRef<T, I> m) {
	using namespace sparse_ldlt::util;

	I* mi = m.row_indices().ptr();
	T* mx = m.values().ptr();

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

	I* hi = h.row_indices().ptr();
	T* hx = h.values().ptr();

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

	I* hi = h.row_indices().ptr();
	T* hx = h.values().ptr();

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

		auto _a_infty_norm = stack.make_new(veg::Tag<T>{}, n);
		auto _c_infty_norm = stack.make_new(veg::Tag<T>{}, n);
		auto _h_infty_norm = stack.make_new(veg::Tag<T>{}, n);

		// norm_infty of each column of A (resp. C), i.e.,
		// each row of AT (resp. CT)
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
		using namespace sparse_ldlt::util;
		for (usize j = 0; j < usize(n_eq); ++j) {
			T a_row_norm = 0;
			qp.AT.to_eigen();
			usize col_start = qp.AT.col_start(j);
			usize col_end = qp.AT.col_end(j);

			for (usize p = col_start; p < col_end; ++p) {
				T aji = ATx[p];
				a_row_norm = std::max(a_row_norm, aji);
			}

			delta(n + j) = T(1) / (machine_eps + sqrt(a_row_norm));
		}

		for (usize j = 0; j < usize(n_in); ++j) {
			T c_row_norm = 0;
			usize col_start = qp.CT.col_start(j);
			usize col_end = qp.CT.col_end(j);

			for (usize p = col_start; p < col_end; ++p) {
				T cji = CTx[p];
				c_row_norm = std::max(c_row_norm, cji);
			}

			delta(n + n_eq + j) = T(1) / (machine_eps + sqrt(c_row_norm));
		}

		// normalize A and C
		qp.AT = delta.head(n).asDiagonal() * qp.AT *
		        delta.segment(n, n_eq).asDiagonal();
		qp.CT = delta.head(n).asDiagonal() * qp.CT * delta.tail(n_in).asDiagonal();
		// normalize vectors
		qp.g.to_eigen().array() *= delta.head(n).array();
		qp.b.to_eigen().array() *= delta.segment(n, n_eq).array();
		qp.l.to_eigen().array() *= delta.tail(n_in).array();
		qp.u.to_eigen().array() *= delta.tail(n_in).array();

		// normalize H
		switch (sym) {
		case Symmetry::LOWER: {
			for (usize j = 0; j < usize(n); ++j) {
				usize col_start = qp.H.col_start(j);
				usize col_end = qp.H.col_end(j);
				T delta_j = delta(j);

				if (col_end > col_start) {
					usize p = col_end;
					while (true) {
						--p;
						usize i = zero_extend(Hi[p]);
						if (i < j) {
							break;
						}
						Hx[p] *= delta_j * delta(i);
					}
				}
			}
			break;
		}
		case Symmetry::UPPER: {
			for (usize j = 0; j < usize(n); ++j) {
				usize col_start = qp.H.col_start(j);
				usize col_end = qp.H.col_end(j);
				T delta_j = delta(j);

				for (usize p = col_start; p < col_end; ++p) {
					usize i = zero_extend(Hi[p]);
					if (i > j) {
						break;
					}
					Hx[p] *= delta_j * delta(i);
				}
			}
			break;
		}
		}

		// additional normalization
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

	void scale_qp_in_place(
			QpViewMut<T, I> qp, VectorViewMut<T> tmp_delta_preallocated) {
		delta.setOnes();
		tmp_delta_preallocated.to_eigen().setZero();
		c = detail::ruiz_scale_qp_in_place(
				{ldlt::from_eigen, delta},
				tmp_delta_preallocated,
				logger_ptr,
				qp,
				epsilon,
				max_iter,
				sym);
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

} // namespace preconditionner
} // namespace sparse
} // namespace qp

#endif /* end of include guard INRIA_LDLT_SOLVER_SPARSE_HPP_YHQF6TYWS */
