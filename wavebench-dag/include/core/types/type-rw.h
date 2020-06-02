#ifndef __TYPE_RW_TYPES_CORE_H__
#define __TYPE_RW_TYPES_CORE_H__
#include <type_traits>

#include "../macros/definitions.h"
#ifdef CUDA_SUPPORT_COREQ
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

#include "../meta/meta.h"
#include "enum-definitions.h"
#include "fundamental-types.h"

namespace __core__ {
namespace __type__ {
template <ReadMode read_mode=normal_read,typename T=void,enable_IT<(read_mode==read_only)&&is_vendor_ro_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address) {
#if defined(__CUDA_ARCH__)
	return __ldg(address);
#else
	return *address;
#endif
}
template <ReadMode read_mode=normal_read,typename T=void,enable_IT<(read_mode==read_only)&&(!is_vendor_ro_CE<T>())&&is_read_only_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address) {
#if defined(__CUDA_ARCH__)
	ro_equivalent_T<T> data=__ldg((ro_equivalent_T<T>*)address);
	return *(T*)(&data);
//	T data;
//	asm("ld.global.nc.v2.f32 {%0,%1}, [%2] ;":"=f"(data[0]),"=f"(data[1]):"l"(address));
//	return data;
#else
	return *address;
#endif
}
template <ReadMode read_mode=normal_read,typename T=void,enable_IT<(read_mode==read_only)&&(!is_read_only_CE<T>())> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address) {
	return *address;
}
template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int,enable_IT<(read_mode==read_only)&&is_vendor_ro_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address,int_T indx) {
#if defined(__CUDA_ARCH__)
	return __ldg(address+indx);
#else
	return address[indx];
#endif
}
template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int,enable_IT<(read_mode==read_only)&&(!is_vendor_ro_CE<T>())&&is_read_only_CE<T>()> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address,int_T indx) {
#if defined(__CUDA_ARCH__)
	ro_equivalent_T<T> data=__ldg((ro_equivalent_T<T>*)(address+idx));
	return *(T*)(&data);
#else
	return address[indx];
#endif
}
template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int,enable_IT<(read_mode==read_only)&&(!is_read_only_CE<T>())> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(const T* __restrict__ address,int_T indx) {
	return *(address+indx);
}

template <ReadMode read_mode=normal_read,typename T=void,enable_IT<read_mode==normal_read> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(T *address) {
	return *address;
}
template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int,enable_IT<read_mode==normal_read> = 0> __forceinline__ __optimize__ __host_device__
T read_memory(T *address,int_T indx) {
	return *(address+indx);
}

template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int> __forceinline__ __forceflatten__ __optimize__ __host_device__
T read_memory(T *address,int_T yindx,int_T xindx,int_T pitch) {
	return read_memory<read_mode>(((T*)(((uchar*)address)+yindx*pitch))+xindx);
}
template <ReadMode read_mode=normal_read,typename T=void,typename int_T=int> __forceinline__ __forceflatten__ __optimize__ __host_device__
T read_memory(T *address,int_T zindx,int_T yindx,int_T xindx,int_T pitch,int_T slice_pitch) {
	return read_memory<read_mode>(((T*)(((uchar*)address)+yindx*pitch+zindx*slice_pitch))+xindx);
}
}
}
#endif
