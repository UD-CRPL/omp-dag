#ifndef __CONSTEXPR_META_CORE_H__
#define __CONSTEXPR_META_CORE_H__

#include <type_traits>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

namespace __core__ {
namespace __meta__ {
//****************************************************************************************************************
//****************************************************************************************************************
//*						Arithmetic and logical functions
//****************************************************************************************************************
//boolean functions
template <bool expr> __HOST__ __DEVICE__ constexpr bool not_CE(){
	return !expr;
}
template <bool V,bool...VV> __HOST__ __DEVICE__ constexpr typename std::enable_if<(sizeof...(VV)==0),bool>::type and_CE() {
	return V;
}
template <bool V,bool...VV> __HOST__ __DEVICE__ constexpr typename std::enable_if<(sizeof...(VV)>=1),bool>::type and_CE() {
	return V&&(and_CE<VV...>());
}
template <bool V,bool...VV> __HOST__ __DEVICE__ constexpr typename std::enable_if<(sizeof...(VV)==0),bool>::type or_CE() {
	return V;
}
template <bool V,bool...VV> __HOST__ __DEVICE__ constexpr typename std::enable_if<(sizeof...(VV)>=1),bool>::type or_CE() {
	return V||(or_CE<VV...>());
}

template <typename T,T u,T v> __HOST__ __DEVICE__ constexpr bool eq_CE() {
	return u==v;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool eq_CE(const T u,const T v) {
	return u==v;
}
template <typename T,T u,T v> __HOST__ __DEVICE__ constexpr bool neq_CE() {
	return u!=v;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool neq_CE(const T u,const T v) {
	return u!=v;
}
template <typename V,typename U> __HOST__ __DEVICE__ constexpr bool lowerThan_CE(const V A,const U B) {
	return A<B;
}
template <typename V,typename U> __HOST__ __DEVICE__ constexpr bool greaterThan_CE(const V A,const U B) {
	return A>B;
}

//numeric functions
template <typename T,T A,T B> __HOST__ __DEVICE__ constexpr T min_CE() {
	return (A>B)?B:A;
}
template <typename T,T A,T B> __HOST__ __DEVICE__ constexpr T max_CE() {
	return (A<B)?B:A;
}
template <typename T> __HOST__ __DEVICE__ constexpr T min_CE(const T A,const T B) {
	return (A>B)?B:A;
}
template <typename T> __HOST__ __DEVICE__ constexpr T max_CE(const T A,const T B) {
	return (A<B)?B:A;
}

template <typename T> __HOST__ __DEVICE__ constexpr T abs_CE(const T x) {
	return x>=0?x:-x;
}

//arithmetic functions
template <typename T> __HOST__ __DEVICE__ constexpr T __gcd_CE__(const T x,const T y) {
	return (y==0)?x:__gcd_CE__(y,x%y);
}
template <typename T> __HOST__ __DEVICE__ constexpr T gcd_CE(const T x,const T y) {
	return (x>y)?__gcd_CE__(x,y):__gcd_CE__(y,x);
}

template <typename T> __HOST__ __DEVICE__ constexpr T pow_CE(const T x,unsigned int N) {
	unsigned int i=N;
	T r=N<=0?1:x;
	while(i-->1)
		r=r*x;
	return r;
}
template <typename T> __HOST__ __DEVICE__ constexpr unsigned int log2_CE(const T x) {
	unsigned int i=0;
	T tmp=x;
	while(tmp>1) {
		++i;
		tmp=tmp/2;
	}
	return i;
}

template <typename T> __HOST__ __DEVICE__ constexpr bool is_pow2_CE(const T x) {
	typedef typename std::make_unsigned<T>::type unsigned_T;
	return ((unsigned_T)x)&&((((unsigned_T)x)&(((unsigned_T)x)-1U))==0U);
}

template <typename T> __HOST__ __DEVICE__ constexpr T nextPow2_CE(const T x) {
	return 1<<static_cast<unsigned int>(log2_CE(x)+1);
}


//****************************************************************************************************************
//****************************************************************************************************************
//*						Type functions
//****************************************************************************************************************
template <typename T,typename U> __HOST__ __DEVICE__ constexpr bool is_same_CE() {
	return std::is_same<T,U>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_ptr_CE() {
	return std::is_pointer<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_ref_CE() {
	return std::is_reference<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_floating_point_CE() {
	return std::is_floating_point<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_fp_CE() {
	return std::is_floating_point<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_numeric_CE() {
	return std::is_arithmetic<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_integer_CE() {
	return std::is_integral<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_signed_CE() {
	return std::is_signed<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_unsigned_CE() {
	return std::is_unsigned<T>::value;
}
template <typename T> __HOST__ __DEVICE__ constexpr bool is_primitive_CE() {
	return is_numeric_CE<T>()||is_ptr_CE<T>();
}

template <typename T,typename int_T=std::size_t> __HOST__ __DEVICE__ constexpr typename std::enable_if<is_same_CE<T,void>(),int_T>::type __sizeof__() {
	return 0;
}
template <typename T,typename int_T=std::size_t> __HOST__ __DEVICE__ constexpr typename std::enable_if<!is_same_CE<T,void>(),int_T>::type __sizeof__() {
	return sizeof(T);
}

template <typename T> constexpr __HOST__ __DEVICE__ bool is_read_only_CE() {
	return (sizeof(T)==alignof(T))&&(sizeof(T)<=16);
}

template <typename T0> __forceinline__  constexpr bool contains_type_CE() {
	return false;
}
template <typename T0,typename T1,typename...TN> __forceinline__  constexpr typename std::enable_if<(sizeof...(TN)==0),bool>::type contains_type_CE() {
	return is_same_CE<T0,T1>();
}
template <typename T0,typename T1,typename...TN> __forceinline__  constexpr typename std::enable_if<(sizeof...(TN)>0),bool>::type contains_type_CE() {
	return is_same_CE<T0,T1>()||contains_type_CE<T0,TN...>();
}

//****************************************************************************************************************
//****************************************************************************************************************
//*						Floating point functions
//****************************************************************************************************************
template <typename T> __attribute__((optimize(0))) __HOST__ __DEVICE__ constexpr bool is_nan_CE(const T x) {
	return x!=x;
}

template <typename T> using __fp_BT=typename std::conditional<(sizeof(T)>4),unsigned long long int,unsigned int>::type;
template <typename T> using __ui_FT=typename std::conditional<(sizeof(T)>4),double,float>::type;
template <typename T> __HOST__ __DEVICE__ constexpr __fp_BT<T> __fp_sign_CE(T x) {
	typedef __fp_BT<T> int_T;
	return x>0?0:(static_cast<int_T>(1)<<(sizeof(T)*8-1));
}
template <typename T> __HOST__ __DEVICE__ constexpr int __fp_exponent_CE__(T x) {
	int exp=0;
	while(((x>2.)||(x<1.))&&(x!=0)) {
		if(x>2.) {
			x=x/2.;
			++exp;
		}
		if(x<1.) {
			x=x*2.;
			--exp;
		}
	}
	return exp;
}
template <typename T> __HOST__ __DEVICE__ constexpr __fp_BT<T> __fp_exponent_CE(T x,const int bias)
{
	typedef __fp_BT<T> int_T;
	return static_cast<int_T>(__fp_exponent_CE__(abs_CE(x))+bias);
}
template <typename T> __HOST__ __DEVICE__ constexpr __fp_BT<T> __fp_mantissa_scale_CE(T x,int exp,const int matissa_bits) {
	typedef __fp_BT<T> int_T;
	x=abs_CE(x);
	while((exp>matissa_bits)||(exp<matissa_bits)) {
		if(exp>matissa_bits) {
			x=x/2.;
			--exp;
		}
		if(exp<matissa_bits) {
			x=x*2.;
			++exp;
		}
	}
	return static_cast<int_T>(x);
}
template <typename T> __HOST__ __DEVICE__ constexpr __fp_BT<T> __fp_mantissa_CE(T x,const int matissa_bits) {
	typedef __fp_BT<T> int_T;
	x=abs_CE(x);
	int_T mask=(static_cast<int_T>(1)<<matissa_bits)-1;
	return __fp_mantissa_scale_CE(x,__fp_exponent_CE__(x),matissa_bits)&mask;
}
template <typename T> __HOST__ __DEVICE__ constexpr __fp_BT<T> __fp2ui_CE(T x,const int matissa_bits) {
	if(x==0) return 0;
	int bias=pow_CE(2,(sizeof(T)*8-(matissa_bits+1)-1))-1;
	return __fp_mantissa_CE(x,matissa_bits)|__fp_sign_CE(x)|(__fp_exponent_CE(x,bias)<<matissa_bits);
}
__HOST__ __DEVICE__ constexpr unsigned int float_to_uint_CE(const float x) {
	return is_nan_CE(x)?0:__fp2ui_CE(x,23);
}
__HOST__ __DEVICE__ constexpr unsigned long long int double_to_ull_CE(const double x) {
	return is_nan_CE(x)?0:__fp2ui_CE(x,52);
}

template <typename T> __HOST__ __DEVICE__ constexpr __ui_FT<T> __ui_fp_sign_CE(T x) {
	return ((x>>(sizeof(T)*8-1))==0)?1.:-1.;
}
template <typename T> __HOST__ __DEVICE__ constexpr int __ui_fp_exponent_CE(T x,const int matissa_bits,const int bias) {
	return static_cast<int>(static_cast<int>(x>>matissa_bits)&((1<<(sizeof(T)*8-(matissa_bits+1)))-1))-bias;
}
template <typename T> __HOST__ __DEVICE__ constexpr T __ui_fp_matissa_CE(T x,const int matissa_bits) {
	return (x&((static_cast<T>(1)<<matissa_bits)-1))|(static_cast<T>(1)<<matissa_bits);
}
template <typename T> __HOST__ __DEVICE__ constexpr T __ui_fp_matissa_scale_CE(T x,int exp,const int matissa_bits) {
	while((exp>matissa_bits)||(exp<matissa_bits)) {
		if(exp>matissa_bits) {
			x=x*2.;
			--exp;
		}
		if(exp<matissa_bits) {
			x=x/2.;
			++exp;
		}
	}
	return x;
}
template <typename T> __HOST__ __DEVICE__ constexpr __ui_FT<T> __ui2fp_CE(T x,const int matissa_bits) {
	typedef __ui_FT<T> FT;
	if(x==0) return 0;
	int bias=pow_CE(2,(sizeof(T)*8-(matissa_bits+1)-1))-1;
	return __ui_fp_sign_CE(x)*__ui_fp_matissa_scale_CE(static_cast<FT>(__ui_fp_matissa_CE(x,matissa_bits)),__ui_fp_exponent_CE(x,matissa_bits,bias),matissa_bits);
}
__HOST__ __DEVICE__ constexpr float uint_to_float_CE(const unsigned int x) {
	return __ui2fp_CE(x,23);
}
__HOST__ __DEVICE__ constexpr double ull_to_double_CE(const unsigned long long int x) {
	return __ui2fp_CE(x,52);
}

//****************************************************************************************************************
//****************************************************************************************************************
//*						CUDA type functions
//****************************************************************************************************************
template <typename T> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE() {
	return false;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<char>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<unsigned char>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<short>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<unsigned short>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<int>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<unsigned int>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<long>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<unsigned long>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<long long>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<unsigned long long>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<float>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<double>() {
	return true;
}
#if defined(CUDA_SUPPORT_COREQ)
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<int2>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<uint2>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<int4>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<uint4>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<float2>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<float4>() {
	return true;
}
template <> constexpr __HOST__ __DEVICE__ bool is_vendor_ro_CE<double2>() {
	return true;
}
#endif
}
}
#endif
