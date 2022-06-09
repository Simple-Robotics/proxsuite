#ifndef VEG_MEMORY_BUFFER_HPP_5HP6NE8ZS
#define VEG_MEMORY_BUFFER_HPP_5HP6NE8ZS

#include "veg/vec.hpp"

namespace veg {
namespace cereal {
struct MemoryBuffer {
private:
	Vec<mem::byte> buf;
	struct Pos {
		usize _;
		Pos() = default;
		~Pos() = default;
		Pos(Pos const&) = default;
		auto operator=(Pos const&) -> Pos& = default;
		Pos(Pos&& rhs) noexcept : _{rhs._} { rhs._ = 0; }
		auto operator=(Pos&& rhs) noexcept -> Pos& {
			auto tmp = rhs._;
			rhs._ = 0;
			_ = tmp;
			return *this;
		}
	} pos = {};

public:
	MemoryBuffer() = default;

	void append_bytes(void const* address, usize len) noexcept {
		buf.reserve(buf.len() + isize(len));
		std::memcpy(buf.raw_ref()->end, address, len);
		buf.raw_mut(unsafe)->end += len;
	}
	void pop_front_bytes(Unsafe /*unsafe*/, void* address, usize len) noexcept {
		std::memcpy(address, buf.ptr() + pos._, len);
		pos._ += len;
	}
};
} // namespace cereal
} // namespace veg

#endif /* end of include guard VEG_MEMORY_BUFFER_HPP_5HP6NE8ZS */
