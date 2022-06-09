#ifndef VEG_FMT_HPP_GQU8XFRUS
#define VEG_FMT_HPP_GQU8XFRUS

#include "veg/type_traits/primitives.hpp"
#include "veg/internal/terminate.hpp"
#include "veg/type_traits/constructible.hpp"
#include "veg/ref.hpp"
#include "veg/type_traits/primitives.hpp"
#include "veg/internal/prologue.hpp"

#include <cstring>
#include <cstdio>

namespace veg {
namespace _detail {
auto snprintf1(char* out, isize n, unsigned type, void* arg) VEG_ALWAYS_NOEXCEPT -> isize;
struct String final {
	isize indent_level = 0;
	struct layout {
		char* ptr;
		isize len;
		isize cap;
	} self = {};

	~String();
	String() = default;
	VEG_INLINE String(String&& other) VEG_ALWAYS_NOEXCEPT : self{other.self} {
		other.self = layout{nullptr, 0, 0};
	}
	String(String const&) = delete;
	auto operator=(String const&) -> String& = delete;
	auto operator=(String&&) -> String& = delete;

	void resize(isize new_len) noexcept;
	void reserve(isize new_cap) noexcept;
	void insert(isize pos, char const* data, isize len) noexcept;
	void insert_newline(isize pos) noexcept;
	void fprintln(std::FILE* f) const noexcept;

	VEG_NODISCARD VEG_INLINE auto data() const noexcept -> char* {
		return self.ptr;
	}
	VEG_NODISCARD VEG_INLINE auto size() const noexcept -> isize {
		return self.len;
	}
	template <usize N>
	VEG_INLINE void append_literal(_detail::NativeChar8 const (&data)[N]) {
		insert(size(), reinterpret_cast<char const*>(&data[0]), isize{N} - 1);
	}
	VEG_INLINE void append_ln() { insert_newline(size()); }
};
} // namespace _detail

namespace fmt {
template <typename T>
struct Debug;

using Buffer = _detail::String;

using BufferMut = Buffer&;
} // namespace fmt
namespace _detail {
namespace _fmt {

struct DbgStructScope {
	DbgStructScope(DbgStructScope const&) = delete;
	DbgStructScope(DbgStructScope&&) = delete;
	auto operator=(DbgStructScope const&) -> DbgStructScope& = delete;
	auto operator=(DbgStructScope&&) -> DbgStructScope& = delete;

	fmt::BufferMut out;
	DbgStructScope(fmt::BufferMut out_) : out(VEG_FWD(out_)) {
		out.append_literal(u8"{");
		++out.indent_level;
	}
	~DbgStructScope() {
		--out.indent_level;
		out.append_ln();
		out.append_literal(u8"}");
	}
};

using BufferMut = veg::fmt::BufferMut;

inline void to_string_impl(BufferMut out, unsigned type, void* arg) {
	isize n = _detail::snprintf1(nullptr, 0, type, arg) + 1;

	isize old_size = out.size();
	out.resize(out.size() + n);
	_detail::snprintf1(out.data() + old_size, n, type, arg);
	out.resize(old_size + n - 1);
}

struct dbg_i {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		auto _ = static_cast<long long signed>(arg.get());
		_fmt::to_string_impl(VEG_FWD(out), 0U, &_);
	}
};
struct dbg_u {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		auto _ = static_cast<long long unsigned>(arg.get());
		_fmt::to_string_impl(VEG_FWD(out), 1U, &_);
	}
};
struct dbg_p {
	template <typename T>
	static void to_string(BufferMut out, Ref<T*> arg) {
		auto* _ = const_cast<void*>(static_cast<void const volatile*>(arg.get()));
		_fmt::to_string_impl(VEG_FWD(out), 2U, _);
	}
};
struct dbg_pf {
	template <typename Ret, typename... Args>
	static void to_string(BufferMut out, Ref<Ret (*)(Args...)> arg) {
		auto* _ = reinterpret_cast<void*>(arg.get());
		_fmt::to_string_impl(VEG_FWD(out), 2U, _);
	}
};
struct dbg_f {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		auto _ = static_cast<long double>(arg.get());
		_fmt::to_string_impl(VEG_FWD(out), 3U + 4U * sizeof(T), &_);
	}
};
struct dbg_b {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		out.insert(out.size(), arg.get() ? "true" : "false", arg.get() ? 4 : 5);
	}
};

template <bool>
struct dbg_g_impl {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> /*arg*/) {
		out.append_literal(u8"{?}");
	}
};

template <>
struct dbg_g_impl<true> {

	template <typename T>
	using Access = typename _detail::member_extract_access<T>::Type;

	template <typename T, typename... Bases, usize... Is, typename... Members>
	static void to_string_impl(
			BufferMut out,
			Ref<T> arg,
			SimpleITuple<_meta::integer_sequence<usize, Is...>, Members Bases::*...>
					member_ptrs) {

		out.insert(
				out.size(),
				reinterpret_cast<char const*>(Access<T>::class_name_ptr()),
				Access<T>::class_name_len());
		out.append_literal(u8" ");
		{
			_fmt::DbgStructScope _{VEG_FWD(out)};
			VEG_EVAL_ALL(
					(void(_.out.append_ln()),
			     void(_.out.insert(
							 _.out.size(),
							 reinterpret_cast<char const*>(
									 Access<T>::member_name_ptrs()._[Is]),
							 Access<T>::member_name_lens()._[Is])),
			     void(_.out.append_literal(u8": ")),
			     void(fmt::Debug<Members>::to_string(
							 VEG_FWD(_.out),
							 ref(arg.get().*
			             (static_cast<SimpleLeaf<Is, Members Bases::*> const&>(
												member_ptrs)
			                  .inner)))),
			     void(_.out.append_literal(u8","))

			         ));
		}
	}

	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		to_string_impl(VEG_FWD(out), arg, Access<T>::member_pointers());
	}
};

struct dbg_g {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		dbg_g_impl<_detail::member_extract_access<T>::value>::to_string(
				VEG_FWD(out), arg);
	}
};

struct EnumName {
	char const* ptr;
	usize len;
};

template <typename T, T __veglib_variable_name> // NOLINT
auto enum_name_impl() noexcept -> EnumName {
	auto const& str = __PRETTY_FUNCTION__;
	auto const& needle = "__veglib_variable_name";

	usize len = sizeof(str) - 1;
	usize needle_len = sizeof(needle) - 1;
	if (!(len > needle_len)) {
		_detail::terminate();
	}

	usize pos = 0;
	while (std::memcmp(str + pos, needle, needle_len) != 0) {
		++pos;
		if (!(pos != (len - needle_len))) {
			_detail::terminate();
		}
	}
	pos += needle_len + 3;
	if (!(pos < len)) {
		_detail::terminate();
	}

	return {
			str + pos,
			len - pos - 1,
	};
}

template <typename T, T x>
auto enum_name() noexcept -> EnumName {
	static const auto cached = _detail::_fmt::enum_name_impl<T, x>();
	return cached;
}

template <typename T, usize... Is>
auto enum_name_runtime_impl(meta::index_sequence<Is...> /*tag*/, T n) noexcept
		-> EnumName {
	using FnType = EnumName() VEG_CPP17(noexcept);
	constexpr FnType* fn_ptrs[] = {
			(&_fmt::enum_name<T, T(Is)>)...,
	};
	auto idx = usize(n);
	if (!(idx < sizeof...(Is))) {
		_detail::terminate();
	}

	return fn_ptrs[idx]();
}

template <typename T>
auto enum_name_runtime(T n) noexcept -> EnumName {
	return _fmt::enum_name_runtime_impl(
			meta::make_index_sequence<usize(T::ENUM_END)>{}, n);
}

struct dbg_e {
	template <typename T>
	static void to_string(BufferMut out, Ref<T> arg) {
		EnumName name = _fmt::enum_name_runtime(arg.get());
		out.insert(out.size(), name.ptr, isize(name.len));
	}
};

template <typename T>
using choose_dbg = meta::if_t<
		VEG_CONCEPT(same<T, bool>),
		dbg_b,
		meta::if_t<
				VEG_CONCEPT(signed_integral<T>),
				dbg_i,
				meta::if_t<
						VEG_CONCEPT(unsigned_integral<T>),
						dbg_u,
						meta::if_t<
								VEG_CONCEPT(floating_point<T>),
								dbg_f,
								meta::if_t<
										VEG_CONCEPT(constructible<bool, T>),
										dbg_b,
										meta::if_t<
												VEG_CONCEPT(enum_type<T>),
												dbg_e,
												dbg_g>>>>>>;

} // namespace _fmt
} // namespace _detail

namespace fmt {
template <typename T>
struct Debug : _detail::_fmt::choose_dbg<T> {};
template <typename T>
struct Debug<T*> : _detail::_fmt::dbg_p {};
template <>
struct Debug<decltype(nullptr)> {
	static void to_string(BufferMut out, Ref<decltype(nullptr)> /*arg*/) {
		_detail::_fmt::dbg_p::to_string(
				VEG_FWD(out), ref(static_cast<void*>(nullptr)));
	}
};
template <typename Ret, typename... Args>
struct Debug<Ret (*)(Args...)> : _detail::_fmt::dbg_pf {};

namespace nb {
struct dbg_to {
	template <typename T>
	VEG_INLINE void operator()(BufferMut out, Ref<T> x) const
			VEG_NOEXCEPT_IF(false) {
		Debug<T>::to_string(VEG_FWD(out), x);
	}
};
} // namespace nb
VEG_NIEBLOID(dbg_to);
} // namespace fmt

namespace _detail {
namespace _fmt {
struct DbgRef {
	template <typename T>
	static void to_string(fmt::BufferMut out, Ref<Ref<T>> arg) {
		fmt::dbg_to(VEG_FWD(out), arg.get());
	}
};
struct DbgMut {
	template <typename T>
	static void to_string(fmt::BufferMut out, Ref<RefMut<T>> arg) {
		fmt::dbg_to(VEG_FWD(out), arg.get().as_const());
	}
};
} // namespace _fmt
} // namespace _detail

namespace fmt {
template <typename T>
struct Debug<Ref<T>> : _detail::_fmt::DbgRef {};
template <typename T>
struct Debug<RefMut<T>> : _detail::_fmt::DbgMut {};
} // namespace fmt

namespace _detail {

inline void dbg_prefix(
		RefMut<String> str,
		unsigned line,
		char const* file,
		char const* fn) noexcept {
	auto& out = str.get();

	if (line != 0 && file != nullptr && fn != nullptr) {
		auto file_len = isize(std::strlen(file));
		auto fn_len = isize(std::strlen(fn));

		out.append_literal(u8"[");
		out.insert(1, fn, fn_len);
		out.append_literal(u8":");
		out.insert(1 + fn_len + 1, file, file_len);
		out.append_literal(u8":");
		fmt::Debug<unsigned>::to_string(out, nb::ref{}(line));
		out.append_literal(u8"] ");
	}
}
} // namespace _detail

namespace nb {
struct dbg_to {
	template <typename T>
	auto operator()( //
			std::FILE* f,
			T&& arg,
			unsigned line =
#if VEG_HAS_BUILTIN(__builtin_LINE)
					__builtin_LINE(),
#else
					0,
#endif

			char const* file =
#if VEG_HAS_BUILTIN(__builtin_FILE)
					__builtin_FILE(),
#else
					nullptr,
#endif

			char const* fn =
#if VEG_HAS_BUILTIN(__builtin_FUNCTION)
					__builtin_FUNCTION()
#else
					nullptr
#endif

	) const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		_detail::String out;

		_detail::dbg_prefix(mut{}(out), line, file, fn);
		fmt::Debug<uncvref_t<T>>::to_string(out, nb::ref{}(arg));
		out.fprintln(f);
		return T(VEG_FWD(arg));
	}
};

struct dbg {
	template <typename T>
	auto operator()( //
			T&& arg,
			unsigned line =
#if VEG_HAS_BUILTIN(__builtin_LINE)
					__builtin_LINE(),
#else
					0,
#endif

			char const* file =
#if VEG_HAS_BUILTIN(__builtin_FILE)
					__builtin_FILE(),
#else
					nullptr,
#endif

			char const* fn =
#if VEG_HAS_BUILTIN(__builtin_FUNCTION)
					__builtin_FUNCTION()
#else
					nullptr
#endif

	) const VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_movable<T>)) -> T {
		return dbg_to{}(stderr, VEG_FWD(arg), line, file, fn);
	}
};
} // namespace nb
VEG_NIEBLOID(dbg_to);
VEG_NIEBLOID(dbg);
} // namespace veg

#include "veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_FMT_HPP_GQU8XFRUS */
