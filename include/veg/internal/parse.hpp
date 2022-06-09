#ifndef VEG_PARSE_HPP_JMZVOEG1S
#define VEG_PARSE_HPP_JMZVOEG1S

#include "veg/box.hpp"
#include "veg/vec.hpp"
#include "veg/uwunion.hpp"
#include "veg/option.hpp"
#include "veg/util/unreachable.hpp"
#include <cassert>

namespace veg {
namespace _detail {
namespace type_parse {
VEG_TAG(from_literal, FromLiteral);
struct StrView {
	struct Inner {
		char const* begin;
		isize len;
	} _{};

	StrView() = default;

	constexpr StrView(FromRawParts /*tag*/, Inner inner) noexcept : _{inner} {}

	template <usize N>
	constexpr StrView(FromLiteral /*tag*/, char const (&literal)[N]) noexcept
			: _{&literal[0], N - 1} {}

	VEG_NODISCARD constexpr auto ptr() const noexcept -> char const* {
		return _.begin;
	}
	VEG_NODISCARD constexpr auto len() const noexcept -> isize {
		return isize(_.len);
	}
	VEG_NODISCARD auto split_at(isize pos) const noexcept
			-> Tuple<StrView, StrView> {
		return (
				assert(pos <= _.len),
				Tuple<StrView, StrView>{
						tuplify,
						{from_raw_parts, {_.begin, pos}},
						{from_raw_parts, {_.begin + pos, _.len - pos}},
				});
	}
	VEG_NODISCARD
	auto substr(isize idx, isize len) const noexcept -> StrView {
		return VEG_ASSERT_ALL_OF(
							 (idx <= this->len()), //
							 (idx + len <= this->len())),
		       StrView{from_raw_parts, {_.begin + idx, len}};
	}
	VEG_NODISCARD auto head(isize len) const noexcept -> StrView {
		return substr(0, len);
	}
	VEG_NODISCARD auto tail(isize len) const noexcept -> StrView {
		return assert(len <= this->len()), substr(this->len() - len, len);
	}
	VEG_NODISCARD auto skip_leading(isize n) const noexcept -> StrView {
		return substr(n, len() - n);
	}

	VEG_NODISCARD
	auto begins_with(StrView other) const noexcept -> bool {
		return (other.len() <= len()) &&
		       (std::memcmp(ptr(), other.ptr(), usize(other.len())) == 0);
	}
	VEG_NODISCARD
	auto ends_with(StrView other) const noexcept -> bool {
		return (other.len() <= len()) && (std::memcmp(
																					ptr() + (len() - other.len()),
																					other.ptr(),
																					usize(other.len())) == 0);
	}
	VEG_NODISCARD auto eq(StrView other) const noexcept -> bool {
		return (other.len() == len()) &&
		       (std::memcmp(ptr(), other.ptr(), usize(len())) == 0);
	}
	VEG_NODISCARD auto ltrim(char c) const noexcept -> StrView {
		isize pos = 0;
		while (true) {
			if (pos == len() || ptr()[pos] != c) {
				break;
			}
			++pos;
		}
		return split_at(pos)[1_c];
	}
	VEG_NODISCARD auto rtrim(char c) const noexcept -> StrView {
		isize pos = len();
		while (true) {
			if (pos == 0 || (ptr()[pos - 1] != c)) {
				break;
			}
			--pos;
		}
		return split_at(pos)[0_c];
	}
	VEG_NODISCARD auto trim(char c) const noexcept -> StrView {
		return ltrim(c).rtrim(c);
	}
};

inline auto operator==(StrView a, StrView b) noexcept -> bool {
	return a.eq(b);
}

struct Entity;
using BoxedEntity =
		Box<Entity,
        mem::SystemAlloc,
        mem::DtorAvailable::yes_nothrow,
        mem::CopyAvailable::no>;
using VecEntity =
		Vec<Entity,
        mem::SystemAlloc,
        mem::DtorAvailable::yes_nothrow,
        mem::CopyAvailable::no>;

enum struct CvQual : unsigned char {
	NONE,
	CONST,
	VOLATILE,
	CONST_VOLATILE,
	ENUM_END,
};
enum struct RefQual : unsigned char {
	LVALUE,
	RVALUE,
	POINTER,
	ENUM_END,
};

struct EntityName {
	StrView name;
	VEG_REFLECT(EntityName, name);
};

struct TemplatedEntity {
	BoxedEntity tpl;
	VecEntity args;
	VEG_REFLECT(TemplatedEntity, tpl, args);
};
struct NestedEntity {
	VecEntity components;
	VEG_REFLECT(NestedEntity, components);
};

struct Pointer {
	BoxedEntity entity;
	RefQual ref_qual;
	VEG_REFLECT(Pointer, entity, ref_qual);
};
struct Array {
	BoxedEntity entity;
	BoxedEntity size;
	VEG_REFLECT(Array, entity, size);
};
struct Function {
	BoxedEntity return_type;
	VecEntity args;
	VEG_REFLECT(Function, return_type, args);
};

struct Entity {
	using Uwunion = veg::Uwunion<
			EntityName,
			TemplatedEntity,
			NestedEntity,

			Pointer,
			Array,
			Function>;

	Uwunion kind;
	CvQual cv_qual;
	VEG_REFLECT(Entity, kind, cv_qual);
};

inline auto operator==(Entity const& lhs, Entity const& rhs) noexcept -> bool;

#define REFLECT_EQ(Class)                                                      \
	inline auto operator==(Class const& lhs, Class const& rhs) noexcept->bool {  \
		return cmp::reflected_eq(lhs, rhs);                                        \
	}

REFLECT_EQ(EntityName);
REFLECT_EQ(TemplatedEntity);
REFLECT_EQ(NestedEntity);
REFLECT_EQ(Pointer);
REFLECT_EQ(Array);
REFLECT_EQ(Function);
REFLECT_EQ(Entity);

enum struct TokenKind : unsigned char {
	UNARY_OP,
	BINARY_OP,
	AMBIGUOUS,
	OPEN_DELIM,
	CLOSE_DELIM,
	COMPOSITE_PRIMITIVE,
	IDENT,
	ENUM_END,
};
struct Token {
	StrView str;
	TokenKind kind;
	VEG_REFLECT(Token, str, kind);
};

struct FunctionDecl {
	Option<Entity> return_type;
	Entity full_name;
	Vec<Entity> args;
	CvQual cv_qual;
	bool is_static;
	Vec<Tuple<Entity, Entity>> dependent_expansions;

	VEG_REFLECT(
			FunctionDecl,
			return_type,
			full_name,
			args,
			cv_qual,
			is_static,
			dependent_expansions);
};

inline auto
operator==(FunctionDecl const& lhs, FunctionDecl const& rhs) noexcept -> bool {
	return veg::cmp::reflected_eq(lhs, rhs);
}

auto parse_function_decl(StrView str) noexcept -> FunctionDecl;
auto greedy_parse_nestable_entity(StrView str) noexcept
		-> Tuple<Entity, StrView>;
auto greedy_parse_entity(StrView str) noexcept -> Tuple<Entity, StrView>;
auto greedy_parse_nested_entity(StrView str) noexcept -> Tuple<Entity, StrView>;
void strip_discard_1st(RefMut<Entity> e_mut) noexcept;
void recurse_strip_discard_1st(RefMut<Entity> e_mut) noexcept;
} // namespace type_parse
} // namespace _detail

template <>
struct fmt::Debug<_detail::type_parse::StrView> {
	static void
	to_string(fmt::Buffer& out, Ref<_detail::type_parse::StrView> str_ref) {
		out.reserve(str_ref->len() + 2);
		isize n = out.size();
		// TODO: escape sequences
		char quotes = '\"';
		auto str = str_ref.get();
		out.insert(n, &quotes, 1);
		out.insert(n + 1, str.ptr(), str.len());
		out.insert(n + 1 + str.len(), &quotes, 1);
	}
};

} // namespace veg

#endif /* end of include guard VEG_PARSE_HPP_JMZVOEG1S */
