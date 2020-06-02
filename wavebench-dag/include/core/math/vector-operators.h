#ifndef __VECTOR_OPERATIONS_MATH_CORE_CUH__
#define __VECTOR_OPERATIONS_MATH_CORE_CUH__

#include <cmath>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"
#include "../macros/functions.h"
#include <math.h>

#include "../meta/meta.h"
#include "../types/types.h"
#include "arithmetic.h"
#include "elemental-functions.h"
#include "vector-norms.h"

#include "vector-arithmetic.h"

namespace __core__ {
namespace __math__ {
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__> struct vector_add {
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u) {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u) {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u,const X alpha) {
        return __add__<ARVT,ARM>(v,u,alpha);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void,typename Y=void> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u,const X alpha,const Y beta) {
        return __add__<ARVT,ARM>(v,u,alpha,beta);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto add(const VVT& v,const U u) {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto add(const VVT& v,const UVT& u) {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void> static inline __forceflatten__ __optimize__ __host_device__
    auto add(const VVT& v,const UVT& u,const X alpha) {
        return __add__<ARVT,ARM>(v,u,alpha);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void,typename Y=void> static inline __forceflatten__ __optimize__ __host_device__
    auto add(const VVT& v,const UVT& u,const X alpha,const Y beta) {
        return __add__<ARVT,ARM>(v,u,alpha,beta);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u) const {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u) const {
        return __add__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u,const X alpha) const {
        return __add__<ARVT,ARM>(v,u,alpha);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename X=void,typename Y=void>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u,const X alpha,const Y beta) const {
        return __add__<ARVT,ARM>(v,u,alpha,beta);
    }
};
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__> struct vector_sub {
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u) {
        return __sub__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u) {
        return __sub__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto sub(const VVT& v,const U u) {
        return __sub__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto sub(const VVT& v,const UVT& u) {
        return __sub__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u) const {
        return __sub__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u) const {
        return __sub__<ARVT,ARM>(v,u);
    }
};
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__> struct vector_mul {
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u) {
        return __mul__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u) {
        return __mul__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto mul(const VVT& v,const U u) {
        return __mul__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto mul(const VVT& v,const UVT& u) {
        return __mul__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u) const {
        return __mul__<ARVT,ARM>(v,u);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u) const {
        return __mul__<ARVT,ARM>(v,u);
    }
};
template <typename RVT=void,FastMathMode FM=__default_fast_math_mode__,RoundingMode RM=__default_rounding_mode__> struct vector_div {
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u) {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u) {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto div(const VVT& v,const U u) {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto div(const VVT& v,const UVT& u) {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename U=void,enable_IT<!is_vector_CE<U>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u) const {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
    template <typename ARVT=RVT,FastMathMode AFM=FM,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,enable_IT<is_vector_CE<UVT>()> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u) const {
        return __div__<ARVT,AFM,ARM>(v,u);
    }
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__> struct vector_ma {
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename W=void,enable_IT<(!is_vector_CE<U>())&&(!is_vector_CE<W>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u,const W w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename W=void,enable_IT<(is_vector_CE<UVT>())&&(!is_vector_CE<W>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u,const W w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename WVT=void,enable_IT<(!is_vector_CE<U>())&&(is_vector_CE<WVT>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const U u,const WVT w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename WVT=void,enable_IT<(is_vector_CE<UVT>())&&(is_vector_CE<WVT>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v,const UVT& u,const WVT& w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename W=void,enable_IT<(!is_vector_CE<U>())&&(!is_vector_CE<W>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto ma(const VVT& v,const U u,const W w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename W=void,enable_IT<(is_vector_CE<UVT>())&&(!is_vector_CE<W>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto ma(const VVT& v,const UVT& u,const W w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename WVT=void,enable_IT<(!is_vector_CE<U>())&&(is_vector_CE<WVT>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto ma(const VVT& v,const U u,const WVT w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename WVT=void,enable_IT<(is_vector_CE<UVT>())&&(is_vector_CE<WVT>())> = 0> static inline __forceflatten__ __optimize__ __host_device__
    auto ma(const VVT& v,const UVT& u,const WVT& w) {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename W=void,enable_IT<(!is_vector_CE<U>())&&(!is_vector_CE<W>())> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u,const W w) const {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename W=void,enable_IT<(is_vector_CE<UVT>())&&(!is_vector_CE<W>())> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u,const W w) const {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename U=void,typename WVT=void,enable_IT<(!is_vector_CE<U>())&&(is_vector_CE<WVT>())> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const U u,const WVT w) const {
        return __ma__<ARVT,ARM>(v,u,w);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void,typename UVT=void,typename WVT=void,enable_IT<(is_vector_CE<UVT>())&&(is_vector_CE<WVT>())> = 0>  inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v,const UVT& u,const WVT& w) const {
        return __ma__<ARVT,ARM>(v,u,w);
    }
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************
template <typename RVT=void,RoundingMode RM=__default_rounding_mode__> struct vector_rcp {
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void> static inline __forceflatten__ __optimize__ __host_device__
    auto fn(const VVT& v) {
        return __rcp__<ARVT,ARM>(v);
    }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void> static inline __forceflatten__ __optimize__ __host_device__
    auto rcp(const VVT& v) {
         return __rcp__<ARVT,ARM>(v);
     }
    template <typename ARVT=RVT,RoundingMode ARM=RM,typename VVT=void> inline __forceflatten__ __optimize__ __host_device__
    auto operator()(const VVT& v) const {
         return __rcp__<ARVT,ARM>(v);
     }
};

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************

template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ auto operator+(V v,Vector<U,1,AU> u) {
	return v+u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ auto operator-(V v,Vector<U,1,AU> u) {
	return v-u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ auto operator*(V v,Vector<U,1,AU> u) {
	return v*(u.x);
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ auto operator/(V v,Vector<U,1,AU> u) {
	return v/u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ auto operator%(V v,Vector<U,1,AU> u) {
	return v%u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator<(V v,Vector<U,1,AU> u) {
	return v<u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator<=(V v,Vector<U,1,AU> u) {
	return v<=u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator>(V v,Vector<U,1,AU> u) {
	return v>u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator>=(V v,Vector<U,1,AU> u) {
	return v>=u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator==(V v,Vector<U,1,AU> u) {
	return v==u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator!=(V v,Vector<U,1,AU> u) {
	return v!=u.x;
}
template<typename V,typename U,uint AU,enable_T<is_integer_CE<V>()&&is_integer_CE<U>(),int> =0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto operator&(V v,Vector<U,1,AU> u) {
	return v&u.x;
}
template<typename V,typename U,uint AU,enable_T<is_integer_CE<V>()&&is_integer_CE<U>(),int> =0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto operator|(V v,Vector<U,1,AU> u) {
	return v|u.x;
}
template<typename V,typename U,uint AU,enable_T<is_integer_CE<V>()&&is_integer_CE<U>(),int> =0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto operator^(V v,Vector<U,1,AU> u) {
	return v^u.x;
}
template<typename V,typename U,uint AU,enable_T<is_integer_CE<V>()&&is_integer_CE<U>(),int> =0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto operator<<(V v,Vector<U,1,AU> u) {
	return v<<u.x;
}
template<typename V,typename U,uint AU,enable_T<is_integer_CE<V>()&&is_integer_CE<U>(),int> =0> __forceinline__ __forceflatten__ __optimize__ __host_device__
auto operator>>(V v,Vector<U,1,AU> u) {
	return v>>u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator&&(V v,Vector<U,1,AU> u) {
	return v&&u.x;
}
template<typename V,typename U,uint AU> __forceinline__ __forceflatten__ __optimize__ __host_device__ bool operator||(V v,Vector<U,1,AU> u) {
	return v||u.x;
}

//******************************************************************************************************************************************************************************
//******************************************************************************************************************************************************************************
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator+(const Vector<V,N,VA>& v,const U u) {
	return __add__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator+(const U u,const Vector<V,N,VA>& v) {
	return __add__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,uint UA=0> __optimize__ __forceflatten__ __host_device__
auto operator+(const Vector<V,N,VA>& v,const Vector<U,N,UA>& u) {
	return __add__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator-(const Vector<V,N,VA>& v,const U u) {
	return __sub__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator-(const U u,const Vector<V,N,VA>& v) {
	return __sub__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,uint UA=0> __optimize__ __forceflatten__ __host_device__
auto operator-(const Vector<V,N,VA>& v,const Vector<U,N,UA>& u) {
	return __sub__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator*(const Vector<V,N,VA>& v,const U u) {
	return __mul__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator*(const U u,const Vector<V,N,VA>& v) {
	return __mul__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,uint UA=0> __optimize__ __forceflatten__ __host_device__
auto operator*(const Vector<V,N,VA>& v,const Vector<U,N,UA>& u) {
	return __mul__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,enable_IT<!is_vector_CE<U>()> = 0> __optimize__ __forceflatten__ __host_device__
auto operator/(const Vector<V,N,VA>& v,const U u) {
	return __div__(v,u);
}
template <typename V=void,typename U=void,int N=0,uint VA=0,uint UA=0> __optimize__ __forceflatten__ __host_device__
auto operator/(const Vector<V,N,VA>& v,const Vector<U,N,UA>& u) {
	return __div__(v,u);
}
}
}
#endif
