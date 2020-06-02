#ifndef __SCALAR_OPERATORS_MATH_CORE_H__
#define __SCALAR_OPERATORS_MATH_CORE_H__

#include "../macros/macros.h"
#include "arithmetic.h"
#include "elemental-functions.h"

namespace __core__ {
namespace __math__ {
//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__> struct scalar_add {
	template <RoundingMode ARM=RM,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x,U y) {
		return __add__<ARM,V,U>(x,y);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x,U y) const {
		return __add__<RM,V,U>(x,y);
	}
};
template <RoundingMode RM=__default_rounding_mode__> struct scalar_sub {
	template <RoundingMode ARM=RM,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x,U y) {
		return __sub__<ARM,V,U>(x,y);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x,U y) const {
		return __sub__<RM,V,U>(x,y);
	}
};
template <RoundingMode RM=__default_rounding_mode__> struct scalar_mul {
	template <RoundingMode ARM=RM,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x,U y) {
		return __mul__<ARM,V,U>(x,y);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x,U y) const {
		return __mul__<RM,V,U>(x,y);
	}
};
template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__> struct scalar_div {
	template <FastMathMode AFM=FM,RoundingMode ARM=RM,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x,U y) {
		return __div__<AFM,ARM,V,U>(x,y);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x,U y) const {
		return __div__<FM,RM,V,U>(x,y);
	}
};
template <RoundingMode RM=__default_rounding_mode__> struct scalar_ma {
	template <RoundingMode ARM=RM,typename T=void,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(T x,V y,U z) {
		return __ma__<ARM,T,V,U>(x,y,z);
	}
	template <typename T=void,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(T x,V y,U z) const {
		return __ma__<RM,T,V,U>(x,y,z);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template <RoundingMode RM=__default_rounding_mode__> struct scalar_rcp {
	template <RoundingMode ARM=RM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __rcp__<ARM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __rcp__<RM,V>(x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

struct scalar_floor {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __floor__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __floor__<V>(x);
	}
};
struct scalar_ceil {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __ceil__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __ceil__<V>(x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

struct scalar_abs {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __abs__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __abs__<V>(x);
	}
};
struct scalar_sign {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __sign__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __sign__<V>(x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

struct scalar_max {
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V y,U x) {
		return __max__<V,U>(y,x);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V y,U x) const {
		return __max__<V,U>(y,x);
	}
};
struct scalar_min {
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V y,U x) {
		return __min__<V,U>(y,x);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V y,U x) const {
		return __min__<V,U>(y,x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template <FastMathMode FM=__default_fast_math_mode__> struct scalar_sin {
	template <FastMathMode AFM=FM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __sin__<AFM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __sin__<FM,V>(x);
	}
};
template <FastMathMode FM=__default_fast_math_mode__> struct scalar_cos {
	template <FastMathMode AFM=FM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __cos__<AFM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __cos__<FM,V>(x);
	}
};
template <FastMathMode FM=__default_fast_math_mode__> struct scalar_tan {
	template <FastMathMode AFM=FM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __tan__<AFM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __tan__<FM,V>(x);
	}
};

struct scalar_asin {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __asin__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __asin__<V>(x);
	}
};
struct scalar_acos {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __acos__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __acos__<V>(x);
	}
};
struct scalar_atan {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __atan__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __atan__<V>(x);
	}
};
struct scalar_atan2 {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V y,V x) {
		return __atan2__<V>(y,x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V y,V x) const {
		return __atan2__<V>(y,x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template <FastMathMode FM=__default_fast_math_mode__> struct scalar_exp {
	template <FastMathMode AFM=FM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __exp__<AFM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __exp__<FM,V>(x);
	}
};
template <FastMathMode FM=__default_fast_math_mode__> struct scalar_pow {
	template <FastMathMode AFM=FM,typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x,U y) {
		return __pow__<AFM,V,U>(x,y);
	}
	template <typename V=void,typename U=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x,U y) const {
		return __pow__<V,U,FM>(x,y);
	}
};
template <FastMathMode FM=__default_fast_math_mode__> struct scalar_log {
	template <FastMathMode AFM=FM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __log__<AFM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __log__<FM,V>(x);
	}
};

template <RoundingMode RM=__default_rounding_mode__> struct scalar_pow2 {
	template <RoundingMode ARM=RM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __pow2__<ARM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __pow2__<RM,V>(x);
	}
};
template <RoundingMode RM=__default_rounding_mode__> struct scalar_pow3 {
	template <RoundingMode ARM=RM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __pow3__<ARM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __pow3__<RM,V>(x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template <FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__> struct scalar_sqrt {
	template <FastMathMode AFM=FM,RoundingMode ARM=RM,typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __sqrt__<AFM,ARM,V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __sqrt__<FM,RM,V>(x);
	}
};
struct scalar_rsqrt {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __rsqrt__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __rsqrt__<V>(x);
	}
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************
#ifdef CUDA_SUPPORT_COREQ
struct scalar_erf {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __erf__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __erf__<V>(x);
	}
};

struct scalar_normcdf {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __normcdf__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __normcdf__<V>(x);
	}
};
struct scalar_normcdfinv {
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ static auto fn(V x) {
		return __normcdfinv__<V>(x);
	}
	template <typename V=void> __optimize__ __forceinline__ __forceflatten__ __host_device__ auto operator()(V x) const {
		return __normcdfinv__<V>(x);
	}
};
#endif
}
}
#endif
