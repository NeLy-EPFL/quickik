// Minimal stand-ins for the `rust::` runtime-support types the cxx-generated
// fastik.h uses (rust::Box, rust::Slice, rust::Str, rust::Vec, rust::Opaque,
// rust::Error). The real definitions are template/macro-heavy plumbing
// (placement-new allocators, name-mangled extern "C" thunks) that Doxygen's
// parser can't handle -- see docs/build.sh, which substitutes this stub in
// their place before running Doxygen. Hidden from the rendered output via
// Doxyfile's EXCLUDE_SYMBOLS; only used so the real fastik:: signatures that
// reference them still parse.
namespace rust {

class Error {};

class Str {
public:
  Str() = default;
};

class Opaque {};

template <typename T> class Box {
public:
  const T &operator*() const;
  const T *operator->() const;
};

template <typename T> class Slice {
public:
  Slice(T *, ::std::size_t);
};

template <typename T> class Vec {
public:
  ::std::size_t size() const;
  const T &operator[](::std::size_t) const;
};

} // namespace rust
