#ifndef __COMPILER_MACROS_CORE_H__
#define __COMPILER_MACROS_CORE_H__

#include "definitions.h"

#ifndef __align__
#define __align__(alignment) __attribute__((aligned(alignment)))
#endif

#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif

#ifndef __optimize__
#define __optimize__  __attribute__((optimize(3)))
#endif

#ifndef __forceflatten__
#define __forceflatten__ __attribute__((flatten))
#endif

#if defined(CUDA_SUPPORT_COREQ)
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif

#define __host_device__ __HOST__ __DEVICE__

#define __attr_opfi__ __optimize__ __forceinline__
#define __attr_ophd__ __optimize__ __host_device__
#define __attr_opff__ __optimize__ __forceflatten__
#define __attr_fihd__ __forceinline__ __host_device__
#define __attr_ffhd__ __forceflatten__ __host_device__
#define __attr_fiff__ __forceinline__ __forceflatten__
#define __attr_opfihd__ __optimize__ __forceinline__ __host_device__
#define __attr_opffhd__ __optimize__ __forceflatten__ __host_device__
#define __attr_opfiff__ __optimize__ __forceinline__ __forceflatten__
#define __attr_fiffhd__ __forceinline__ __forceflatten__ __host_device__
#define __attr_opfiffhd__ __optimize__ __forceinline__ __forceflatten__ __host_device__

#endif
