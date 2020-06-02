#ifndef __ARITHMETIC_MATH_CORE_H__
#define __ARITHMETIC_MATH_CORE_H__

#include <cmath>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"
#include "../macros/functions.h"
#include <math.h>

#include "../types/types.h"
#include "../meta/meta.h"

namespace __core__ {
namespace __math__ {

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************
template <RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__
higher_PT<V,U> __add__(const V x,const U y) {
	return x+y;
}
template <> __forceinline__ __optimize__ __host_device__ float __add__<RN,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fadd_rn(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __add__<RD,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fadd_rd(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __add__<RU,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fadd_ru(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __add__<RZ,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fadd_rz(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __add__<RN,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dadd_rn(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __add__<RD,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dadd_rd(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __add__<RU,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dadd_ru(x,y);
#else
	return x+y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __add__<RZ,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dadd_rz(x,y);
#else
	return x+y;
#endif
}

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__
higher_PT<V,U> __sub__(const V x,const U y) {
	return x-y;
}
template <> __forceinline__ __optimize__ __host_device__ float __sub__<RN,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fsub_rn(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sub__<RD,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fsub_rd(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sub__<RU,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fsub_ru(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __sub__<RZ,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fsub_rz(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sub__<RN,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dsub_rn(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sub__<RD,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dsub_rd(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sub__<RU,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dsub_ru(x,y);
#else
	return x-y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __sub__<RZ,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dsub_rz(x,y);
#else
	return x-y;
#endif
}

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__
higher_PT<V,U> __mul__(const V x,const U y) {
	return x*y;
}
template <> __forceinline__ __optimize__ __host_device__ float __mul__<RN,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fmul_rn(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __mul__<RD,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fmul_rd(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __mul__<RU,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fmul_ru(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __mul__<RZ,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fmul_rz(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __mul__<RN,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dmul_rn(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __mul__<RD,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dmul_rd(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __mul__<RU,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dmul_ru(x,y);
#else
	return x*y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __mul__<RZ,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __dmul_rz(x,y);
#else
	return x*y;
#endif
}

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__
higher_PT<V,U> __div_fp__(const V x,const U y) {
	return x/y;
}
template <> __forceinline__ __optimize__ __host_device__ float __div_fp__<RN,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fdiv_rn(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __div_fp__<RD,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fdiv_rd(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __div_fp__<RU,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fdiv_ru(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __div_fp__<RZ,float,float,0>(const float x,const float y) {
#if defined(__CUDA_ARCH__)
	return __fdiv_rz(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __div_fp__<RN,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __ddiv_rn(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __div_fp__<RD,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __ddiv_rd(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __div_fp__<RU,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __ddiv_ru(x,y);
#else
	return x/y;
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __div_fp__<RZ,double,double,0>(const double x,const double y) {
#if defined(__CUDA_ARCH__)
	return __ddiv_rz(x,y);
#else
	return x/y;
#endif
}

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************

template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<!(is_same_CE<V,float>()&&is_same_CE<V,U>()&&FM)> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto  __div__(const V x,const U y) {
	return __div_fp__<RM,V,U>(x,y);
}
template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__,typename V=void,typename U=void,enable_IT<is_same_CE<V,float>()&&is_same_CE<V,U>()&&FM> = 0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto  __div__(const V x,const U y) {
#if defined(__CUDA_ARCH__)
	return __fdividef(x,y);
#else
	return x/y;
#endif
}

//*************************************************************************************************************************************************
//*************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__,typename T=void,typename V=void,typename U=void,enable_IT<is_numeric_CE<T>()&&is_numeric_CE<V>()&&is_numeric_CE<U>()> = 0> __forceinline__ __optimize__ __host_device__ 
higher_PT<T,higher_PT<V,U>> __ma__(const T x,const V y,const U z) {
	return (x*y)+z;
}
template <> __forceinline__ __optimize__ __host_device__ float __ma__<RN,float,float,float,0>(const float x,const float y,const float z) {
#if defined(__CUDA_ARCH__)
	return __fmaf_rn(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __ma__<RD,float,float,float,0>(const float x,const float y,const float z) {
#if defined(__CUDA_ARCH__)
	return __fmaf_rd(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __ma__<RU,float,float,float,0>(const float x,const float y,const float z) {
#if defined(__CUDA_ARCH__)
	return __fmaf_ru(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ float __ma__<RZ,float,float,float,0>(const float x,const float y,const float z) {
#if defined(__CUDA_ARCH__)
	return __fmaf_rz(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __ma__<RN,double,double,double,0>(const double x,const double y,const double z) {
#if defined(__CUDA_ARCH__)
	return __fma_rn(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __ma__<RD,double,double,double,0>(const double x,const double y,const double z) {
#if defined(__CUDA_ARCH__)
	return __fma_rd(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __ma__<RU,double,double,double,0>(const double x,const double y,const double z) {
#if defined(__CUDA_ARCH__)
	return __fma_ru(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
template <> __forceinline__ __optimize__ __host_device__ double __ma__<RZ,double,double,double,0>(const double x,const double y,const double z) {
#if defined(__CUDA_ARCH__)
	return __fma_rz(x,y,z);
#else
	return std::fma(x,y,z);
#endif
}
}
}
#endif
