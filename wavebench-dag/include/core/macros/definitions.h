#ifndef __DEFINITIONS_MACROS_CORE_H__
#define __DEFINITIONS_MACROS_CORE_H__

//#define CUDA_SUPPORT_COREQ
#ifdef __NVCC__
#define CUDA_SUPPORT_COREQ
#else
#ifdef CUDA_SUPPORT_COREQ
#undef CUDA_SUPPORT_COREQ
#endif
#endif

#ifdef CUDA_SUPPORT_COREQ
#define __CUDARUNTIMEQ__
#endif

#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
typedef cudaStream_t StreamType;
#else
typedef void* StreamType;
#endif
static constexpr StreamType DEFAULT_STREAM=0;

//type definitions
#define RESTRICT_Q(x) x __restrict__
#define CRESTRICT_Q(x) const x __restrict__

#define C_A(type,x) constant_argument<type,__constant_argument__<type,x>>::value
#define CTA(type,x) constant_argument<type,__constant_argument__<type,template_argument<type>(x)>>

#endif
