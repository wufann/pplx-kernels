#pragma once

#include <hip/hip_runtime.h>

namespace pplx {

void sleepOnStream(double seconds, hipStream_t stream);

}
