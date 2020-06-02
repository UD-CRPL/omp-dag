#ifndef __TYPE_CAST_CORE_H__
#define __TYPE_CAST_CORE_H__

#include <limits>
#include <type_traits>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"
#include <math.h>

#include "../enum-definitions.h"
#include "enum-definitions.h"

namespace __core__ {
namespace __type__ {
template <typename DT,RoundingMode RM=__default_rounding_mode__,typename ST=void> __optimize__ __forceinline__ __host_device__ DT __cast__(const ST v) {
	return static_cast<DT>(v);
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RN,int>(const int v) {
#if defined(__CUDA_ARCH__)
	return __int2float_rn(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RD,int>(const int v) {
#if defined(__CUDA_ARCH__)
	return __int2float_rd(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RU,int>(const int v) {
#if defined(__CUDA_ARCH__)
	return __int2float_ru(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RZ,int>(const int v) {
#if defined(__CUDA_ARCH__)
	return __int2float_rz(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RN,unsigned int>(const unsigned int v) {
#if defined(__CUDA_ARCH__)
	return __uint2float_rn(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RD,unsigned int>(const unsigned int v) {
#if defined(__CUDA_ARCH__)
	return __uint2float_rd(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RU,unsigned int>(const unsigned int v) {
#if defined(__CUDA_ARCH__)
	return __uint2float_ru(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RZ,unsigned int>(const unsigned int v) {
#if defined(__CUDA_ARCH__)
	return __uint2float_rz(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RN,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2float_rn(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RD,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2float_rd(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RU,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2float_ru(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RZ,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2float_rz(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RN,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2float_rn(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RD,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2float_rd(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RU,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2float_ru(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ float __cast__<float,RZ,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2float_rz(v);
#else
	return static_cast<float>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RN,int>(const int v) {
#if defined(__CUDA_ARCH__)
	return __int2double_rn(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RN,unsigned int>(const unsigned int v) {
#if defined(__CUDA_ARCH__)
	return __uint2double_rn(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RN,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2double_rn(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RD,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2double_rd(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RU,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2double_ru(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RZ,long long>(const long long v) {
#if defined(__CUDA_ARCH__)
	return __ll2double_rz(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RN,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2double_rn(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RD,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2double_rd(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RU,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2double_ru(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ double __cast__<double,RZ,unsigned long long>(const unsigned long long v) {
#if defined(__CUDA_ARCH__)
	return __ull2double_rz(v);
#else
	return static_cast<double>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RN,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2int_rn(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RD,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2int_rd(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RU,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2int_ru(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RZ,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2int_rz(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RN,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2int_rn(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RD,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2int_rd(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RU,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2int_ru(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ int __cast__<int,RZ,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2int_rz(v);
#else
	return static_cast<int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RN,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2uint_rn(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RD,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2uint_rd(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RU,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2uint_ru(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RZ,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2uint_rz(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RN,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2uint_rn(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RD,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2uint_rd(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RU,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2uint_ru(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned int __cast__<unsigned int,RZ,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2uint_rz(v);
#else
	return static_cast<unsigned int>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RN,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ll_rn(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RD,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ll_rd(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RU,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ll_ru(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RZ,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ll_rz(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RN,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ll_rn(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RD,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ll_rd(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RU,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ll_ru(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ long long __cast__<long long,RZ,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ll_rz(v);
#else
	return static_cast<long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RN,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ull_rn(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RD,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ull_rd(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RU,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ull_ru(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RZ,float>(const float v) {
#if defined(__CUDA_ARCH__)
	return __float2ull_rz(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RN,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ull_rn(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RD,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ull_rd(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RU,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ull_ru(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
template <> __optimize__ __forceinline__ __host_device__ unsigned long long __cast__<unsigned long long,RZ,double>(const double v) {
#if defined(__CUDA_ARCH__)
	return __double2ull_rz(v);
#else
	return static_cast<unsigned long long>(v);
#endif
}
}
}
#endif
