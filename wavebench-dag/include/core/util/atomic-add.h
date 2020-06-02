#ifndef __ATOMIC_ADD_UTIL_CORE_H__
#define __ATOMIC_ADD_UTIL_CORE_H__
#ifdef CUDA_SUPPORT_COREQ
#if __CUDA_ARCH__ < 600
#include "../macros/definitions.h"
#include <cuda_runtime.h>

namespace __core__
{
__device__ double atomicAdd(double* address, double val);
}
#endif
#endif
#endif
