#ifndef __VECTOR_TYPES_CORE_H__
#define __VECTOR_TYPES_CORE_H__
#include <type_traits>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"
#include "../macros/functions.h"

#include "../meta/meta.h"
#include "fundamental-types.h"

namespace __core__ {
namespace __type__ {
namespace __type_alignment__ {
template <typename T,typename IT=uint> constexpr uint valignment() {
	return static_cast<IT>(alignof(T));
}
constexpr uint __valignment__(const uint SIZE) {
	return ((SIZE&15U)==0U)?16U:(((SIZE&7U)== 0U)?8U:4U);
}
template <typename T,uint n,typename IT=uint,uint SIZE=n*sizeof(T)> constexpr uint valignment() {
	return 	(is_pow2_CE(SIZE)&&(SIZE<=16))?SIZE:((sizeof(T)>=4)?__valignment__(SIZE):__valignment__(nextPow2_CE(SIZE)));
}
}
//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
// 																	Vector template class
//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
template <typename T,int n,uint alignment=__type_alignment__::valignment<T>()> struct __align__(alignment) Vector {
	static_assert(n>0,"The vector size must be positive.");
	static_assert(is_pow2_CE(alignment),"The alignment must be a power of two.");

	T x[n];

	__attr_opfihd__ T* operator*();

	template <typename IT> __attr_opfihd__ T operator()(IT i) const;
	template <typename IT> __attr_opfihd__ T& operator[](IT i);
	template <typename IT> __attr_opfihd__ const T& operator[](IT i) const;

	inline __attr_opffhd__ Vector<T,n,alignment>& operator=(const T y);
	template <typename V,uint VA> inline __attr_ophd__ explicit operator Vector<V,n,VA>() const;

	template <typename U,uint UA> inline __attr_ophd__ bool operator==(const Vector<U,n,UA> &u);

	template <typename IT> __attr_opfihd__ volatile T& read(IT i) volatile;
	template <typename IT> __attr_opfihd__ const volatile T& read(IT i) const volatile;
};

template <typename T,int n,uint alignment> __attr_opfihd__
T* Vector<T,n,alignment>::operator*() {
	return &x;
}
template <typename T,int n,uint alignment> template <typename IT> __attr_opfihd__
T Vector<T,n,alignment>::operator()(IT i) const {
	return x[i];
}
template <typename T,int n,uint alignment> template <typename IT> __attr_opfihd__
T& Vector<T,n,alignment>::operator[](IT i) {
	return x[i];
}
template <typename T,int n,uint alignment> template <typename IT> __attr_opfihd__
const T& Vector<T,n,alignment>::operator[](IT i) const {
	return x[i];
}
template <typename T,int n,uint alignment> inline __attr_opffhd__
Vector<T,n,alignment>& Vector<T,n,alignment>::operator=(const T y) {
#if defined(__CUDA_ARCH__)
	__unroll_gpu__(int,i,0,n,(T*,T),(T* x,T y),(x,y),(x[i]=y;))
#else
	__unroll_cpu__(int,i,0,n,(T*,T),(T* x,T y),(x,y),(x[i]=y;))
#endif
	return *this;
}
template <typename T,int n,uint alignment> template <typename V,uint VA> inline __attr_ophd__
Vector<T,n,alignment>::operator Vector<V,n,VA>() const {
	typedef Vector<V,n,VA> vec_T;
	Vector<V,n,VA> r;
#if defined(__CUDA_ARCH__)
	__unroll_gpu__(int,i,0,n,(vec_T&,const T*),(vec_T& r,const T* x),(r,x),(r[i]=static_cast<V>(x[i]);))
#else
	__unroll_cpu__(int,i,0,n,(vec_T&,const T*),(vec_T& r,const T* x),(r,x),(r[i]=static_cast<V>(x[i]);))
#endif
	return r;
}
template <typename T,int n,uint alignment> template <typename U,uint UA> inline __attr_ophd__
bool Vector<T,n,alignment>::operator==(const Vector<U,n,UA> &u) {
	for(int i=0;i<n;++i)
		if(x[i]!=u.x[i])
			return false;
	return true;
}
template <typename T,int n,uint alignment> template <typename IT> __attr_opfihd__
volatile T& Vector<T,n,alignment>::read(IT i) volatile {
	return x[i];
}
template <typename T,int n,uint alignment> template <typename IT> __attr_opfihd__
const volatile T& Vector<T,n,alignment>::read(IT i) const volatile {
	return x[i];
}

//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
// 																	Vector specialization class	n=2
//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
template <typename T,uint alignment> struct __align__(alignment) Vector<T,2,alignment> {
	static_assert(is_pow2_CE(alignment),"The alignment size must be a power of two!!!");
	T x[2];

	__attr_opfihd__ T* operator*();

	template <typename IT> __attr_opfihd__ T operator()(IT i) const;
	template <typename IT> __attr_opfihd__ T& operator[](IT i);
	template <typename IT> __attr_opfihd__ const T& operator[](IT i) const;

	__attr_opfiffhd__ Vector& operator=(const T y);
	template <typename V,uint VA> inline __attr_ophd__ explicit operator Vector<V,2,VA>() const;

	template <typename V,uint VA> __attr_opfiffhd__ bool operator<(const Vector<V,2,VA> &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator<=(const Vector<V,2,VA> &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator>(const Vector<V,2,VA> &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator>=(const Vector<V,2,VA> &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator==(const Vector<V,2,VA> &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator!=(const Vector<V,2,VA> &v) const;

	template <typename IT> __attr_opfihd__ volatile T& read(IT i) volatile;
	template <typename IT> __attr_opfihd__ const volatile T& read(IT i) const volatile;
};

template <typename T,uint alignment> __attr_opfihd__
T* Vector<T,2,alignment>::operator*() {
	return &x;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
T Vector<T,2,alignment>::operator()(IT i) const {
	return x[i];
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
T& Vector<T,2,alignment>::operator[](IT i) {
	return x[i];
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
const T& Vector<T,2,alignment>::operator[](IT i) const {
	return x[i];
}
template <typename T,uint alignment> __attr_opfiffhd__
Vector<T,2,alignment> &Vector<T,2,alignment>::operator=(const T y) {
	x[0]=y;
	x[1]=y;
	return *this;
}
template <typename T,uint alignment> template <typename V,uint VA> inline __attr_ophd__
Vector<T,2,alignment>::operator Vector<V,2,VA>() const {
	Vector<V,2,VA> r;
	r.x[0]=static_cast<V>(x[0]);
	r.x[1]=static_cast<V>(x[1]);
	return r;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator<(const Vector<V,2,VA> &v) const {
	return (x[0]<v(0)?true:(x[0]==v(0)?(x[1]<v(1)):false));
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator<=(const Vector<V,2,VA> &v) const {
	return (x[0]<v(0)?true:(x[0]==v(0)?(x[1]<=v(1)):false));
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator>(const Vector<V,2,VA> &v) const {
	return (x[0]>v(0)?true:(x[0]==v(0)?(x[1]>v(1)):false));
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator>=(const Vector<V,2,VA> &v) const {
	return (x[0]>=v(0)?true:(x[0]==v(0)?(x[1]>=v(1)):false));
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator==(const Vector<V,2,VA> &v) const {
	return (x[0]==v(0))&&(x[1]==v(1));
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,2,alignment>::operator!=(const Vector<V,2,VA> &v)const {
	return (x[0]!=v(0))||(x[1]!=v(1));
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
volatile T& Vector<T,2,alignment>::read(IT i) volatile {
	return x[i];
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
const volatile T& Vector<T,2,alignment>::read(IT i) const volatile {
	return x[i];
}

//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
// 																	Vector specialization class	n=1
//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
template <typename T,uint alignment>  struct __align__(alignment) Vector<T,1,alignment> {
	static_assert((alignment>0)&&is_pow2_CE(alignment),"The alignment size must be positive");
	T x;

	template <typename V> __attr_fihd__ Vector(V y);

	template <typename V> __attr_fihd__ T& operator=(V y);

	__attr_opfihd__ T& operator*();

	template <typename IT> __attr_opfihd__ T operator()(IT i= 0) const;
	template <typename IT> __attr_opfihd__ T& operator[](IT i);
	template <typename IT> __attr_opfihd__ const T& operator[](IT i) const;

	template <typename V,typename U=T,enable_IT<is_primitive_CE<V>()&&is_primitive_CE<U>()> = 0> __attr_fihd__ operator V() const;
	template <typename V,uint VA> inline __attr_ophd__ explicit operator Vector<V,1,VA>() const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator+(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator-(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator*(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator/(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator%(Vector<V,1,VA> v) const;
	template <typename V> __attr_opfiffhd__ auto operator+(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator-(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator*(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator/(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator%(V v) const;

	template <typename V,uint VA> __attr_opfiffhd__ bool operator<(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator<=(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator>(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator>=(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator==(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator!=(Vector<V,1,VA> v) const;
	template <typename V> __attr_opfiffhd__ bool operator<(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator<=(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator>(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator>=(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator==(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator!=(V v) const;

	__attr_opfiffhd__ T& operator++();
	__attr_opfiffhd__ T operator++(int);
	__attr_opfiffhd__ T& operator--();
	__attr_opfiffhd__ T operator--(int);

	template <typename V,uint VA> __attr_opfiffhd__ T operator~() const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator&(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator|(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator^(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator<<(Vector<V,1,VA> v) const;
	template <typename V,uint VA> __attr_opfiffhd__ auto operator>>(Vector<V,1,VA> v) const;
	template <typename V> __attr_opfiffhd__ auto operator&(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator|(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator^(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator<<(V v) const;
	template <typename V> __attr_opfiffhd__ auto operator>>(V v) const;

	__attr_opfiffhd__ bool operator!() const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator&&(const Vector &v) const;
	template <typename V,uint VA> __attr_opfiffhd__ bool operator||(const Vector &v) const;
	template <typename V> __attr_opfiffhd__ bool operator&&(V v) const;
	template <typename V> __attr_opfiffhd__ bool operator||(V v) const;

	template <typename IT> __attr_opfihd__ volatile T& read(IT i) const;
	template <typename IT> __attr_opfihd__ const volatile T& read(IT i) const volatile;
};

template <typename T,uint alignment> template <typename V> __attr_fihd__
Vector<T,1,alignment>::Vector(V y) {
	x=y;
}
template <typename T,uint alignment> template <typename V> __attr_fihd__
T& Vector<T,1,alignment>::operator=(V y) {
	x=y;
	return x;
}
template <typename T,uint alignment> __attr_opfihd__
T& Vector<T,1,alignment>::operator*() {
	return x;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
T Vector<T,1,alignment>::operator()(IT i) const {
	return x;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
T& Vector<T,1,alignment>::operator[](IT i) {
	return x;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
const T& Vector<T,1,alignment>::operator[](IT i) const {
	return x;
}
template <typename T,uint alignment> template <typename V,typename U,enable_IT<is_primitive_CE<V>()&&is_primitive_CE<U>()>> __attr_fihd__
Vector<T,1,alignment>::operator V() const {
	return static_cast<V>(x);
}
template <typename T,uint alignment> template <typename V,uint VA> inline __attr_ophd__
Vector<T,1,alignment>::operator Vector<V,1,VA>() const {
	Vector<V,1,VA> r;
	r.x[0]=static_cast<V>(x[0]);
	return r;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator+(Vector<V,1,VA> v) const {
	return x+v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator-(Vector<V,1,VA> v) const {
	return x-v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator*(Vector<V,1,VA> v) const {
	return x*v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator/(Vector<V,1,VA> v) const {
	return x/v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator%(Vector<V,1,VA> v) const {
	return x%v.x;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator+(V v) const {
	return x+v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator-(V v) const {
	return x-v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator*(V v) const {
	return x*v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator/(V v) const {
	return x/v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator%(V v) const {
	return x%v;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator<(Vector<V,1,VA> v) const {
	return x<v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator<=(Vector<V,1,VA> v) const {
	return x<=v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator>(Vector<V,1,VA> v) const {
	return x>v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator>=(Vector<V,1,VA> v) const {
	return x>=v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator==(Vector<V,1,VA> v) const {
	return x==v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator!=(Vector<V,1,VA> v) const {
	return x!=v.x;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator<(V v) const {
	return x<v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator<=(V v) const {
	return x<=v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator>(V v) const {
	return x>v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator>=(V v) const {
	return x>=v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator==(V v) const {
	return x==v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator!=(V v) const {
	return x!=v;
}
template <typename T,uint alignment> __attr_opfiffhd__
T& Vector<T,1,alignment>::operator++() {
	return ++x;
}
template <typename T,uint alignment> __attr_opfiffhd__
T Vector<T,1,alignment>::operator++(int) {
	return x++;
}
template <typename T,uint alignment> __attr_opfiffhd__
T& Vector<T,1,alignment>::operator--() {
	return --x;
}
template <typename T,uint alignment> __attr_opfiffhd__
T Vector<T,1,alignment>::operator--(int) {
	return x--;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
T Vector<T,1,alignment>::operator~() const {
	return ~x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator&(Vector<V,1,VA> v) const {
	return x&v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator|(Vector<V,1,VA> v) const {
	return x|v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator^(Vector<V,1,VA> v) const {
	return x^v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator<<(Vector<V,1,VA> v) const {
	return x<<v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator>>(Vector<V,1,VA> v) const {
	return x>>v.x;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator&(V v) const {
	return x&v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator|(V v) const {
	return x|v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator^(V v) const {
	return x^v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator<<(V v) const {
	return x<<v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
auto Vector<T,1,alignment>::operator>>(V v) const {
	return x>>v;
}
template <typename T,uint alignment> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator!() const {
	return !x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator&&(const Vector &v) const {
	return x&&v.x;
}
template <typename T,uint alignment> template <typename V,uint VA> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator||(const Vector &v) const {
	return x||v.x;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator&&(V v) const {
	return x&&v;
}
template <typename T,uint alignment> template <typename V> __attr_opfiffhd__
bool Vector<T,1,alignment>::operator||(V v) const {
	return x||v;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
volatile T& Vector<T,1,alignment>::read(IT i) const {
	return x;
}
template <typename T,uint alignment> template <typename IT> __attr_opfihd__
const volatile T& Vector<T,1,alignment>::read(IT i) const volatile {
	return x;
}

//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
// 																	Vector types & helpers
//*************************************************************************************************************************************************************************************
//*************************************************************************************************************************************************************************************
template <typename T,int n,uint alignment=(__type_alignment__::valignment<T>())> using make_vector_T=Vector<T,n,alignment>;
template <typename T,int n> using make_vector_AT=Vector<T,n,__type_alignment__::valignment<T,n>()>;
template <typename T> struct is_vector_ST {
	static constexpr bool value=false;
};
template <template <typename,int,uint> class CT,typename T,int n,uint S> struct is_vector_ST<CT<T,n,S>> {
	static constexpr bool value=is_same_ST<CT<T,n,S>,Vector<T,n,S>>::value;
};
template <typename T> __host_device__ constexpr bool is_vector_CE() {
	return is_vector_ST<T>::value;
}
}
}
#endif
