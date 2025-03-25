#pragma once

#include <unistd.h>

#define _ROSE_ASSERT_MSG(msg)                                                                      \
  do {                                                                                             \
    ssize_t ret = write(2, msg, sizeof(msg));                                                      \
    (void)ret;                                                                                     \
  } while (false);

#define _ROSE_STRINGIFY(x) #x
#define _ROSE_EXPAND_AND_STRINGIFY(x) _ROSE_STRINGIFY(x)

#define ROSE_ASSERT(cond, msg)                                                                     \
  if (!(cond)) {                                                                                   \
    _ROSE_ASSERT_MSG(__FILE__ "(" _ROSE_EXPAND_AND_STRINGIFY(__LINE__) "): " msg "\n");            \
    __builtin_trap();                                                                              \
  }

#define ROSE_UNREACHABLE(msg)                                                                      \
  _ROSE_ASSERT_MSG(__FILE__ "(" _ROSE_EXPAND_AND_STRINGIFY(__LINE__) "): " msg "\n");              \
  __builtin_trap();

namespace pplx {

template <typename T> T ceil_div(T x, T y) { return (x + y - 1) / y; }

template <typename T> T round_up(T x, T y) { return ceil_div<T>(x, y) * y; }

} // namespace pplx
