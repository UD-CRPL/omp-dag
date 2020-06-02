#ifndef __REDUCE_FUNCTIONAL_H__
#define __REDUCE_FUNCTIONAL_H__

#if defined(CUDA_SUPPORT_COREQ)
#include "reduce-functions.cuh"
#endif

#include "array/reduce-array.h"

namespace __core__ {
namespace __functional__ {
using namespace __reduce__;
}
}

#endif
