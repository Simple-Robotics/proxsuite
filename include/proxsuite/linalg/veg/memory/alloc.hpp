#ifndef VEG_ALLOC_HPP_TAWYRUICS
#define VEG_ALLOC_HPP_TAWYRUICS

#include "proxsuite/linalg/veg/ref.hpp"
#include "proxsuite/linalg/veg/type_traits/constructible.hpp"
#include "proxsuite/linalg/veg/type_traits/assignable.hpp"
#include "proxsuite/linalg/veg/internal/typedefs.hpp"
#include "proxsuite/linalg/veg/internal/macros.hpp"
#include "proxsuite/linalg/veg/memory/placement.hpp"
#include "proxsuite/linalg/veg/type_traits/alloc.hpp"

#include <cstddef> // std::max_align_t
#include <cstdlib> // std::{malloc, free, realloc}, ::{aligned_alloc, free}
#ifndef __APPLE__
#include <malloc.h> // ::malloc_usable_size
#else
#include <AvailabilityMacros.h>
#include <malloc/malloc.h>
#define malloc_usable_size malloc_size
#endif
#include "proxsuite/linalg/veg/internal/prologue.hpp"

namespace proxsuite {
namespace linalg {
namespace veg {

#ifdef __APPLE__
namespace alignment {

#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
VEG_INLINE void*
aligned_alloc(std::size_t alignment, std::size_t size)
{
  return std::aligned_alloc(alignment, size);
}
#elif MAC_OS_X_VERSION_MIN_REQUIRED >= 1090
VEG_INLINE void*
aligned_alloc(std::size_t alignment, std::size_t size)
{
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
  void* p;
  if (::posix_memalign(&p, alignment, size) != 0) {
    p = 0;
  }
  return p;
}
#endif

} // namespace alignment
#endif

namespace mem {
enum struct CopyAvailable
{
  no,
  yes_maythrow,
  yes_nothrow,
};
enum struct DtorAvailable
{
  no,
  yes_maythrow,
  yes_nothrow,
};
template<typename T>
struct CopyAvailableFor
  : meta::constant<mem::CopyAvailable,
                   (VEG_CONCEPT(nothrow_copyable<T>) &&
                    VEG_CONCEPT(nothrow_copy_assignable<T>))
                     ? CopyAvailable::yes_nothrow
                   : (VEG_CONCEPT(copyable<T>) &&
                      VEG_CONCEPT(copy_assignable<T>))
                     ? CopyAvailable::yes_maythrow
                     : CopyAvailable::no>
{
};
template<typename T>
struct DtorAvailableFor
  : meta::constant<mem::DtorAvailable,
                   VEG_CONCEPT(nothrow_destructible<T>)
                     ? DtorAvailable::yes_nothrow
                     : DtorAvailable::yes_maythrow>
{
};

VEG_INLINE auto
aligned_alloc(usize align, usize size) noexcept -> void*
{
  usize const mask = align - 1;
#if defined(_WIN32)
  return _aligned_malloc((size + mask) & ~mask, align);
#elif defined(__APPLE__)
  return alignment::aligned_alloc(align, (size + mask) & ~mask);
#else
  return std::aligned_alloc(align, (size + mask) & ~mask);
#endif
}

VEG_INLINE void
aligned_free(usize /*align*/, void* ptr) noexcept
{
#ifndef _WIN32
  std::free(ptr);
#else
  _aligned_free(ptr);
#endif
}

struct SystemAlloc
{
  constexpr friend auto operator==(SystemAlloc /*unused*/,
                                   SystemAlloc /*unused*/) noexcept -> bool
  {
    return true;
  }
};
template<>
struct Alloc<SystemAlloc>
{
  static constexpr usize max_base_align = alignof(std::max_align_t);

  VEG_INLINE static void dealloc(RefMut<SystemAlloc> /*alloc*/,
                                 void* ptr,
                                 Layout layout) noexcept
  {
    (layout.align <= max_base_align) ? std::free(ptr)
                                     : mem::aligned_free(layout.align, ptr);
  }
  VEG_NODISCARD VEG_INLINE static auto alloc(RefMut<SystemAlloc> /*alloc*/,
                                             Layout layout) noexcept
    -> mem::AllocBlock
  {
    void* ptr = (layout.align <= max_base_align)
                  ? std::malloc(layout.byte_size)
                  : mem::aligned_alloc(layout.align, layout.byte_size);
    if (HEDLEY_UNLIKELY(ptr == nullptr)) {
      _detail::terminate();
    }
#ifndef _WIN32
    return { ptr, ::malloc_usable_size(ptr) };
#else
    return { ptr, _msize(ptr) };
#endif
  }
  VEG_NODISCARD VEG_NO_INLINE static auto realloc(RefMut<SystemAlloc> /*alloc*/,
                                                  void* ptr,
                                                  Layout layout,
                                                  usize new_size,
                                                  usize copy_size,
                                                  RelocFn reloc) noexcept
    -> mem::AllocBlock
  {
    void* new_ptr; // NOLINT
    bool typical_align = layout.align <= max_base_align;
    bool trivial_reloc = reloc.is_trivial();
    bool use_realloc = typical_align && trivial_reloc;

    if (use_realloc) {
      new_ptr = std::realloc(ptr, new_size);
    } else {
      new_ptr = mem::aligned_alloc(layout.align, new_size);
    }

    if (HEDLEY_UNLIKELY(new_ptr == nullptr)) {
      _detail::terminate();
    }

    if (!use_realloc) {
      reloc(new_ptr, ptr, copy_size);
      mem::aligned_free(layout.align, ptr);
    }
#ifndef _WIN32
    return { new_ptr, ::malloc_usable_size(new_ptr) };
#else
    return { new_ptr, _msize(new_ptr) };
#endif
  }
  VEG_NODISCARD VEG_INLINE auto try_grow_in_place(
    void* /*ptr*/,
    Layout /*layout*/,
    usize /*new_size*/) const noexcept -> bool
  {
    return false;
  }
  VEG_NODISCARD VEG_INLINE static auto grow(RefMut<SystemAlloc> alloc,
                                            void* ptr,
                                            Layout layout,
                                            usize new_size,
                                            RelocFn reloc) noexcept
    -> mem::AllocBlock
  {
    return realloc(
      VEG_FWD(alloc), ptr, layout, new_size, layout.byte_size, reloc);
  }
  VEG_NODISCARD VEG_INLINE static auto shrink(RefMut<SystemAlloc> alloc,
                                              void* ptr,
                                              Layout layout,
                                              usize new_size,
                                              RelocFn reloc) noexcept
    -> mem::AllocBlock
  {
    return realloc(VEG_FWD(alloc), ptr, layout, new_size, new_size, reloc);
  }
};

struct DefaultCloner
{};
template<>
struct Cloner<DefaultCloner>
{
  template<typename T>
  using trivial_clone = meta::bool_constant<VEG_CONCEPT(trivially_copyable<T>)>;

  template<typename T, typename Alloc>
  VEG_INLINE static void destroy(RefMut<DefaultCloner> /*cloner*/,
                                 T* ptr,
                                 RefMut<Alloc> /*alloc*/)
    VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_destructible<T>))
  {
    mem::destroy_at(ptr);
  }
  VEG_TEMPLATE((typename T, typename Alloc),
               requires(VEG_CONCEPT(copyable<T>)),
               VEG_NODISCARD VEG_INLINE static auto clone,
               (/*cloner*/, RefMut<DefaultCloner>),
               (rhs, Ref<T>),
               (/*alloc*/, RefMut<Alloc>))
  VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_copyable<T>))->T { return T(rhs.get()); }
  VEG_TEMPLATE((typename T, typename Alloc),
               requires(VEG_CONCEPT(copyable<T>)),
               VEG_INLINE static void clone_from,
               (/*cloner*/, RefMut<DefaultCloner>),
               (lhs, RefMut<T>),
               (rhs, Ref<T>),
               (/*alloc*/, RefMut<Alloc>))
  VEG_NOEXCEPT_IF(VEG_CONCEPT(nothrow_copy_assignable<T>))
  {
    lhs.get() = rhs.get();
  }
};

VEG_INLINE_VAR(system_alloc, SystemAlloc);
VEG_INLINE_VAR(default_cloner, DefaultCloner);
} // namespace mem

namespace _detail {
namespace _mem {
template<typename A>
struct ManagedAlloc /* NOLINT */
{
  void* data;
  mem::Layout layout;
  RefMut<A> alloc;

  VEG_INLINE ~ManagedAlloc()
  {
    if (data != nullptr) {
      mem::Alloc<A>::dealloc(VEG_FWD(alloc), VEG_FWD(data), VEG_FWD(layout));
    }
  }
};
} // namespace _mem
} // namespace _detail
} // namespace veg
} // namespace linalg
} // namespace proxsuite

#include "proxsuite/linalg/veg/internal/epilogue.hpp"
#endif /* end of include guard VEG_ALLOC_HPP_TAWYRUICS */
