#ifndef __UTIL_MEMORY_CORE_H__
#define __UTIL_MEMORY_CORE_H__

#include <type_traits>

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
#endif

#include "../enum-definitions.h"
#include "../meta/meta.h"

namespace __core__ {
namespace __memory__ {
namespace __memory_private__ {
constexpr MemoryTransferType transfer_type(const DeviceType src_dev,const DeviceType dst_dev) {
	return (src_dev==HOST)?
				((dst_dev==HOST)?CPU_CPU:((dst_dev==DEVICE)?CPU_GPU:CPU_MAN)):
		   ((src_dev==DEVICE)?
				((dst_dev==HOST)?GPU_CPU:((dst_dev==DEVICE)?GPU_GPU:GPU_MAN)):
		        ((dst_dev==HOST)?MAN_CPU:((dst_dev==DEVICE)?MAN_GPU:MAN_MAN)));
}
template <typename memory_ST,typename memory_DT> constexpr MemoryTransferType transfer_type() {
	return transfer_type(memory_ST::location,memory_DT::location);
}
#if defined(__CUDARUNTIMEQ__)
constexpr cudaMemcpyKind memcpykind(const DeviceType src_dev,const DeviceType dst_dev) {
	return (src_dev==HOST)?
				((dst_dev==HOST)?cudaMemcpyHostToHost:((dst_dev==DEVICE)?cudaMemcpyHostToDevice:cudaMemcpyDefault)):
		   ((src_dev==DEVICE)?
				((dst_dev==HOST)?cudaMemcpyDeviceToHost:((dst_dev==DEVICE)?cudaMemcpyDeviceToDevice:cudaMemcpyDefault)):cudaMemcpyDefault);
}
template <typename memory_ST,typename memory_DT> constexpr cudaMemcpyKind memcpykind(){
	return memcpykind(memory_ST::location,memory_DT::location);
}
#endif
template <typename T,typename int_T=std::size_t> __host_device__ constexpr typename std::enable_if<!is_same_CE<T,void>(),int_T>::type __sizeof__() {
	return is_same_CE<void,T>()?1:__core__::__meta__::__sizeof__<T,int_T>();
}
}
template <typename T,enable_IT<std::is_arithmetic<rmPtr_T<T>>::value&&!std::is_pointer<T>::value> = 0> __host_device__ void move(T& x,T& y) {
	x=y;
	y=0;
}
template <typename T,enable_IT<std::is_pointer<T>::value> = 0> __host_device__ void move(T& x,T& y) {
	x=y;
	y=nullptr;
}
template <typename T,enable_IT<!(std::is_arithmetic<rmPtr_T<T>>::value&&!std::is_pointer<T>::value)> = 0> void move(T& x,T&& y) {
	x=std::move(std::forward<T>(y));
}
template <typename T> void delete_object(T* &x) {
	if(x!=nullptr) {
		delete x;
		x=nullptr;
	}
}
template <typename T> void delete_array(T* &x) {
	if(x!=nullptr) {
		delete[] x;
		x=nullptr;
	}
}
template <typename T> void free_memory(T* &x) {
	if(x!=nullptr) {
		free(x);
		x=nullptr;
	}
}
}
}
#endif
