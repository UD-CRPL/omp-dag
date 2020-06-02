#ifndef __TYPES_META_CORE__H__
#define __TYPES_META_CORE__H__

#include <limits>
#include <utility>
#include <type_traits>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

#include "constexpr.h"

namespace __core__ {
namespace __meta__ {
//****************************************************************************************************************
//****************************************************************************************************************
//*						STD library helpers
//****************************************************************************************************************
template <typename T> using addConst_T=const T;
template <typename T> using rmConst_T=typename std::remove_const<T>::type;
template <typename T> using rmPtr_T=typename std::remove_pointer<T>::type;
template <typename T> using unsigned_T=typename std::make_unsigned<T>::type;
template <typename T> using signed_T=typename std::make_signed<T>::type;

template <bool B,typename T=void> using enable_T=typename std::enable_if<B,T>::type;
template <bool B> using enable_IT=typename std::enable_if<B,int>::type;
template <bool B> using enable_VT=typename std::enable_if<B,void>::type;

template <bool B,typename U,typename V> using conditional_T=typename std::conditional<B,U,V>::type;

template <int I,typename T> using tuple_ET=typename std::tuple_element<I,T>::type;

template <typename T> struct __numeric_limits__ {
    static constexpr T eps=std::numeric_limits<T>::epsilon();
    static constexpr T inf=std::numeric_limits<T>::infinity();
    static constexpr T max=std::numeric_limits<T>::max();
    static constexpr T min=std::numeric_limits<T>::lowest();
    static constexpr T max_representable=std::numeric_limits<T>::max();
    static constexpr T min_representable=std::numeric_limits<T>::min();
};

template <typename T> struct is_floating_point_ST {
	static constexpr bool value=is_floating_point_CE<T>();
};
template <typename T> struct is_integer_ST {
	static constexpr bool value=is_integer_CE<T>();
};
template <typename T> struct is_numeric_ST {
	static constexpr bool value=is_numeric_CE<T>();
};
template <typename T,typename U> struct is_same_ST{
	static constexpr bool value=is_same_CE<T,U>();
};

//****************************************************************************************************************
//****************************************************************************************************************
//*						MemberQ helpers
//****************************************************************************************************************
template <typename T> struct has_member_fn_ST {
	template <typename U=T> static int64_t has_member_helper(enable_IT<is_ptr_CE<decltype(&U::fn)*>()> ) {
		return 0;
	}
	template <typename U=T> static int8_t has_member_helper(...) {
		return 0;
	}
	static constexpr bool value=(sizeof(has_member_helper(0))==sizeof(int64_t));
};
template <typename T> struct has_member_value_ST {
	template <typename U=T> static int64_t has_member_helper(enable_IT<is_ptr_CE<decltype(U::value)*>()> ) {
		return 0;
	}
	template <typename U=T> static int8_t has_member_helper(...) {
		return 0;
	}
	static constexpr bool value=(sizeof(has_member_helper(0))==sizeof(int64_t));
};
template <typename T> struct has_member_override_ST {
	template <typename U=T> static int64_t has_member_helper(enable_IT<is_ptr_CE<decltype(U::override)*>()> ) {
		return 0;
	}
	template <typename U=T> static int8_t has_member_helper(...) {
		return 0;
	}
	static constexpr bool value=(sizeof(has_member_helper(0))==sizeof(int64_t));
};

//****************************************************************************************************************
//****************************************************************************************************************
//*						Library helper types
//****************************************************************************************************************
template <typename T,typename U> struct precision_type {
	static_assert(is_numeric_CE<T>()&&is_numeric_CE<U>(),"Used types are not arithmetical types!!!");
	typedef conditional_T<(is_floating_point_CE<T>()||is_floating_point_CE<U>()),
			conditional_T<is_floating_point_CE<T>(),
			conditional_T<is_floating_point_CE<U>(),
			conditional_T<(sizeof(T)>=sizeof(U)),T,U>,T>,U>,
			conditional_T<(sizeof(T)>sizeof(U)),T,
			conditional_T<(sizeof(T)<sizeof(U)),U,
			conditional_T<is_unsigned_CE<T>(),T,U>>>> higher_precision;
	typedef conditional_T<is_same_CE<higher_precision,T>(),U,T> lower_precision;
};
template <typename T,typename U> using higher_PT=typename precision_type<T,U>::higher_precision;
template <typename T,typename U> using lower_PT=typename precision_type<T,U>::lower_precision;

template <typename T> struct read_only_type {
	static_assert(is_read_only_CE<T>(),"T is not read-only readable!!!");
	static constexpr int size=sizeof(T);
#if defined(CUDA_SUPPORT_COREQ)
	typedef conditional_T<size==1,uint8_t,conditional_T<size==2,uint16_t,conditional_T<size==4,uint32_t,conditional_T<size==8,uint64_t,uint4>>>> type;
#else
	struct __attribute__((aligned(16))) __uint128__ {
		uint32_t x,y,z,w;
	};
	typedef __uint128__ uint128_t;
	typedef conditional_T<size==1,uint8_t,conditional_T<size==2,uint16_t,conditional_T<size==4,uint32_t,conditional_T<size==8,uint64_t,uint128_t>>>> type;
#endif
};
template <typename T> using read_only_T=typename read_only_type<T>::type;

//****************************************************************************************************************
//****************************************************************************************************************
//*						Vector-type helpers
//****************************************************************************************************************
template <typename T> struct extract_type {
	typedef T type;
	typedef T scalar_type;
	typedef T vector_type;
	typedef T original_type;
	static constexpr int dim=1;
	static constexpr unsigned int alignment=alignof(T);
};
template <template<typename> class CT,typename T> struct extract_type<CT<T>> {
	typedef T type;
	typedef T scalar_type;
	typedef CT<T> vector_type;
	typedef CT<T> original_type;
	static constexpr int dim=1;
	static constexpr unsigned int alignment=alignof(T);
};
template <template<typename,int,unsigned int> class CT,typename T,int N,unsigned int A> struct extract_type<CT<T,N,A>> {
	typedef T type;
	typedef T scalar_type;
	typedef CT<T,N,A> vector_type;
	typedef CT<T,N,A> original_type;
	static constexpr int dim=N;
	static constexpr unsigned int alignment=A;
};
template <template<typename,unsigned int,unsigned int> class CT,typename T,unsigned int N,unsigned int A> struct extract_type<CT<T,N,A>> {
	typedef T type;
	typedef T scalar_type;
	typedef CT<T,N,A> vector_type;
	typedef CT<T,N,A> original_type;
	static constexpr unsigned int dim=N;
	static constexpr unsigned int alignment=A;
};

template <typename T> using underlying_T=typename extract_type<T>::type;
template <typename T> using basal_T=typename extract_type<T>::type;

template <typename vector_T,typename IT=int> __forceinline__ __HOST__ __DEVICE__ constexpr IT vdim() {
    return static_cast<IT>(extract_type<vector_T>::dim);
}
template <typename vector_T,typename IT=unsigned int> __forceinline__ __HOST__ __DEVICE__ constexpr IT valignment() {
    return static_cast<IT>(extract_type<vector_T>::alignment);
}
template <typename vector_T,typename IT=unsigned int> __forceinline__ __HOST__ __DEVICE__ constexpr IT vsize() {
    return static_cast<IT>(sizeof(vector_T));
}
template <typename vector_T,typename IT=unsigned int> __forceinline__ __HOST__ __DEVICE__ constexpr IT esize() {
    return static_cast<IT>(sizeof(typename extract_type<vector_T>::type));
}

template <typename T0,typename T1,typename...TN> __forceinline__ __HOST__ __DEVICE__ constexpr enable_T<(sizeof...(TN)==0),bool> same_dimensions() {
    return (vdim<T0>()==vdim<T1>());
}
template <typename T0,typename T1,typename...TN> __forceinline__ __HOST__ __DEVICE__ constexpr enable_T<(sizeof...(TN)>0),bool> same_dimensions() {
    return (vdim<T0>()==vdim<T1>())&&same_dimensions<T1,TN...>();
}

//****************************************************************************************************************
//****************************************************************************************************************
//*						Constant arrays & arguments
//****************************************************************************************************************
//constant template arguments
template <typename T> using constant_argument_T=conditional_T<is_floating_point_ST<T>::value,conditional_T<(sizeof(T)>4),unsigned long long int,unsigned int>,T>;
template <typename T> using carg_T=constant_argument_T<T>;

template <typename T,constant_argument_T<T> V> struct __constant_argument__ {
	typedef T value_type;
	typedef constant_argument_T<T> argument_type;
	static constexpr argument_type value=V;
};

template <typename T> __HOST__ __DEVICE__ constexpr constant_argument_T<T> template_argument(const T val) {
	return (constant_argument_T<T>)val;
}
template <> __HOST__ __DEVICE__ constexpr constant_argument_T<float> template_argument(const float val) {
	return float_to_uint_CE(val);
}
template <> __HOST__ __DEVICE__ constexpr constant_argument_T<double> template_argument(const double val) {
	return double_to_ull_CE(val);
}

template <typename T,typename  V> struct constant_argument {
	typedef T type;
	typedef T value_type;
	typedef V argument_type;
	static constexpr T value=V::value;
};
template <typename V> struct constant_argument<float,V> {
    typedef float type;
	typedef float value_type;
    typedef V argument_type;
	static constexpr float value=uint_to_float_CE(V::value);
};
template <typename V> struct constant_argument<double,V> {
    typedef double type;
	typedef double value_type;
    typedef V argument_type;
	static constexpr double value=ull_to_double_CE(V::value);
};

//constant arrays
template <int N,typename...sequence> struct __get_type_sequence__;
template <int N> struct __get_type_sequence__<N> {
    typedef void type;
};
template <typename T,typename...sequence> struct __get_type_sequence__<0,T,sequence...> {
    typedef T type;
};
template <int N,typename T,typename...sequence> struct __get_type_sequence__<N,T,sequence...> {
	static_assert(N>0,"Invalid index!!!");
	static_assert((sizeof...(sequence))>=N,"Invalid index!!!");
	typedef typename __get_type_sequence__<N-1,sequence...>::type type;
};

template <int N,typename...type_sequence> using get_nth_T=typename __get_type_sequence__<N,type_sequence...>::type;
template <typename...T> struct types_array {
    static constexpr int size=sizeof...(T);
    template <int N> using nth_T=get_nth_T<N,T...>;
};

template <int N,typename T,T... sequence> struct __get_sequence_element__;
template <int N,typename T> struct __get_sequence_element__<N,T> {
    static constexpr T value=0;
};
template <typename T,T T0,T... sequence> struct __get_sequence_element__<0,T,T0,sequence...> {
    static constexpr T value=T0;
};
template <int N,typename T,T T0,T... sequence> struct __get_sequence_element__<N,T,T0,sequence...> {
    static constexpr T value=__get_sequence_element__<N-1,T,sequence...>::value;
};

template <typename T,carg_T<T>... N> struct constant_array {
	typedef T type;
	typedef types_array<constant_argument<T,__constant_argument__<T,N>>...> types;
	template <int I> using nth_T=typename types::template nth_T<I>;
	template <int I> using nth_VT=typename nth_T<I>::value_type;

	static constexpr T arr[]={constant_argument<T,__constant_argument__<T,N>>::value...};
	static constexpr T values[]={constant_argument<T,__constant_argument__<T,N>>::value...};
	static constexpr carg_T<T> arr_raw[]={N...};
    static constexpr int size=sizeof...(N);

    template <int n> __forceinline__ __HOST__ __DEVICE__ static constexpr carg_T<T> __get__() {
        return arr_raw[n];
    }
    template <int n> __forceinline__ __HOST__ __DEVICE__ static constexpr T get() {
        return arr[n];
    }
};

template <typename...T> struct constant_variant {
	typedef types_array<T...> types;
    template <int N> using nth_T=get_nth_T<N,T...>;
    template <int N> using nth_VT=typename get_nth_T<N,T...>::value_type;
    static constexpr int size=sizeof...(T);
    template <int N> __forceinline__ __HOST__ __DEVICE__ static constexpr typename get_nth_T<N,T...>::value_type get() {
        return get_nth_T<N,T...>::value;
    }
};
}
}
#endif
