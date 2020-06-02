#ifndef __MEMORY_SET_MEMORY_CORE_H__
#define __MEMORY_SET_MEMORY_CORE_H__

#include <cstring>
#include <chrono>
#include <future>
#include <type_traits>

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

#include "../enum-definitions.h"
#include "../debug/debug.h"
#include "../meta/meta.h"
#include "util.h"

namespace __core__ {
namespace __memory__ {
//linear memory
template <typename memory_T,SyncBehaviorType sync_behavior=SYNC,typename T=void,enable_IT<(memory_T::location==HOST)&&(sync_behavior==SYNC)> = 0>
void __memset__(T *ptr,int value,std::size_t size,int device=0,StreamType stream=(StreamType)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	memset(reinterpret_cast<void*>(ptr),value,size*sod);
}
template <typename memory_T,SyncBehaviorType sync_behavior=SYNC,typename T=void,enable_IT<(memory_T::location==HOST)&&(sync_behavior==ASYNC)> = 0>
std::future<void*> __memset__(T *ptr,int value,std::size_t size,int device=0,StreamType stream=(StreamType)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	return std::async(std::launch::async,memset,reinterpret_cast<void*>(ptr),value,size*sod);
}
#if defined(__CUDARUNTIMEQ__)
template <typename memory_T,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<memory_T::location!=HOST> = 0>
void __memset__(T *ptr,int value,std::size_t size,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemsetAsync(reinterpret_cast<void*>(ptr),value,size*sod,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemset(reinterpret_cast<void*>(ptr),value,size*sod),MEMORY_ERROR);
	}
}
#endif

//2D memory
template <typename memory_T,SyncBehaviorType sync_behavior=ASYNC,enable_IT<memory_T::location==HOST> = 0>
auto __memset2D__(void *ptr,std::size_t pitch,std::size_t xsize,std::size_t ysize,int value,int device=-1,StreamType stream=(StreamType)0) {
	return __memset__<memory_T>(ptr,value,pitch*ysize);
}
#if defined(__CUDARUNTIMEQ__)
template <typename memory_T,SyncBehaviorType sync_behavior=ASYNC,enable_IT<memory_T::location!=HOST> = 0>
void __memset2D__(void *ptr,std::size_t pitch,std::size_t xsize,std::size_t ysize,int value,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemset2DAsync(ptr,pitch,value,xsize,ysize,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemset2D(ptr,pitch,value,xsize,ysize),MEMORY_ERROR);
	}
}
#endif

//3D memory
#if defined(__CUDARUNTIMEQ__)
template <typename memory_T,SyncBehaviorType sync_behavior=ASYNC,enable_IT<memory_T::location==HOST> = 0>
auto __memset3D__(cudaPitchedPtr &ptr,int value,const cudaExtent &extent,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	return __memset__<memory_T>(ptr.ptr,value,ptr.pitch*extent.depth*ptr.ysize);
}
template <typename memory_T,SyncBehaviorType sync_behavior=ASYNC,enable_IT<memory_T::location!=HOST> = 0>
void __memset3D__(cudaPitchedPtr &ptr,int value,const cudaExtent &extent,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemset3DAsync(ptr,value,extent,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemset3D(ptr,value,extent),MEMORY_ERROR);
	}
}
#endif
}
}
#endif
