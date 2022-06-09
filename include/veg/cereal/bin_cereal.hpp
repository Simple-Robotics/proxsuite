#ifndef VEG_CEREAL_HPP_E8ZGU7OLS
#define VEG_CEREAL_HPP_E8ZGU7OLS

#include <veg/internal/macros.hpp>
#include <veg/type_traits/alloc.hpp>
#include <veg/memory/address.hpp>
#include <veg/memory/placement.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace veg {

namespace cereal {
template <typename T>
struct BinCereal;
} // namespace cereal

namespace concepts {
namespace aux {
namespace cereal {
VEG_CONCEPT_EXPR(
		(typename T, typename File),
		(T, File),
		bin_serializable,
		veg::cereal::BinCereal<T>::serialize_to( //
				VEG_DECLVAL(RefMut<File>),
				VEG_DECLVAL(Ref<T>)),
		true);

VEG_CONCEPT_EXPR(
		(typename T, typename File),
		(T, File),
		bin_deserializable_unsafe,
		veg::cereal::BinCereal<T>::unchecked_deserialize_from( //
				unsafe,
				Tag<T>{},
				VEG_DECLVAL(RefMut<File>)),
		VEG_CONCEPT(same<ExprType, T>));
} // namespace cereal
} // namespace aux
VEG_DEF_CONCEPT_CONJUNCTION(
		(typename T, typename File),
		bin_cereal,
		((aux::cereal::, bin_serializable<T, File>),
     (aux::cereal::, bin_deserializable_unsafe<T, File>)));
} // namespace concepts

namespace _detail {
namespace _cereal {
struct Trivial {
	template <typename T, typename File>
	VEG_INLINE static void serialize_to(RefMut<File> f, Ref<T> t) noexcept {
		f.get().append_bytes( //
				static_cast<void const*>(mem::addressof(t.get())),
				sizeof(T));
	}
	template <typename T, typename File>
	VEG_INLINE static auto unchecked_deserialize_from( //
			Unsafe /*unsafe*/,
			Tag<T> /*tag*/,
			RefMut<File> f) noexcept -> T {
		alignas(T) mem::byte buf[sizeof(T)];
		f.get().pop_front_bytes( //
				unsafe,
				static_cast<void*>(&buf[0]),
				sizeof(T));
		return mem::nb::bit_cast<T>{}(buf);
	}
};
} // namespace _cereal
} // namespace _detail

template <>
struct cereal::BinCereal<char> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<bool> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<unsigned char> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<signed char> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<unsigned short> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<signed short> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<unsigned int> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<signed int> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<unsigned long> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<signed long> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<unsigned long long> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<signed long long> : _detail::_cereal::Trivial {};

template <>
struct cereal::BinCereal<float> : _detail::_cereal::Trivial {};
template <>
struct cereal::BinCereal<double> : _detail::_cereal::Trivial {};
} // namespace veg

#endif /* end of include guard VEG_CEREAL_HPP_E8ZGU7OLS */
