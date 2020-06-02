#ifndef __MEMORY_ALLOC_MEMORY_CORE_H__
#define __MEMORY_ALLOC_MEMORY_CORE_H__

#include <cstdlib>
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
//Linear memory
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,HOST)> = 0> __forceinline__
T* __malloc__(std::size_t size,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr=nullptr;
	if(!memory_T::is_pinned)
		ptr=reinterpret_cast<T*>(std::malloc(size*sod));
#ifdef __CUDARUNTIMEQ__
	else {
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		cuda_error(cudaHostAlloc((void **)&(ptr),size*sod,memory_T::host_alloc_F),MEMORY_ERROR);
	}
#endif
	return ptr;
}

#ifdef __CUDARUNTIMEQ__
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,DEVICE)> = 0> __forceinline__
T* __malloc__(std::size_t size,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMalloc((void **)&(ptr),size*sod),MEMORY_ERROR);
	return ptr;
}
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,MANAGED)> = 0> __forceinline__
T* __malloc__(std::size_t size,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMallocManaged((void **)&(ptr),size*sod,memory_T::managed_F),MEMORY_ERROR);
	return ptr;
}
#endif
template <typename memory_T,typename T> __forceinline__
void __malloc__(T* &ptr,std::size_t size,int device=-1) {
	ptr=__malloc__<memory_T,T>(size,device);
}

//2D memory
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,HOST)> = 0> __forceinline__
T* __malloc__(std::size_t& pitch,std::size_t xsize,std::size_t ysize,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr=nullptr;
	pitch=xsize*sod;
	if(!memory_T::is_pinned)
		ptr=reinterpret_cast<T*>(std::malloc(pitch*ysize));

#ifdef __CUDARUNTIMEQ__
	else {
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		cuda_error(cudaHostAlloc((void **)&(ptr),pitch*ysize,memory_T::host_alloc_F),MEMORY_ERROR);
	}
#endif
	return ptr;
}

#ifdef __CUDARUNTIMEQ__
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,DEVICE)> = 0> __forceinline__
T* __malloc__(std::size_t& pitch,std::size_t xsize,std::size_t ysize,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMallocPitch((void **)&(ptr),&pitch,xsize*sod,ysize),MEMORY_ERROR);
	return ptr;
}
template <typename memory_T,typename T=void,enable_IT<eq_CE(memory_T::location,MANAGED)> = 0> __forceinline__
T* __malloc__(std::size_t& pitch,std::size_t xsize,std::size_t ysize,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	T* ptr;
	pitch=xsize*sod;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMallocManaged((void **)&(ptr),pitch*ysize,memory_T::managed_F),MEMORY_ERROR);
	return ptr;
}
#endif
template <typename memory_T,typename T> __forceinline__
void __malloc__(T* &ptr,std::size_t &pitch,std::size_t xsize,std::size_t ysize,int device=-1) {
	ptr=__malloc__<memory_T,T>(pitch,xsize,ysize,device);
}

//3D memory
#ifdef __CUDARUNTIMEQ__
template <typename memory_T,enable_IT<eq_CE(memory_T::location,HOST)> = 0>
cudaPitchedPtr __malloc__(const cudaExtent& extent,int device=-1) {
	cudaPitchedPtr ptr;
	ptr.pitch=extent.width;
	ptr.xsize=extent.width;
	ptr.ysize=extent.height;
	if(!memory_T::is_pinned)
		ptr.ptr=std::malloc(extent.depth*extent.height*extent.width);
	else {
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		cuda_error(cudaHostAlloc((void **)&(ptr.ptr),extent.depth*extent.height*extent.width,memory_T::host_alloc_F),MEMORY_ERROR);
	}
	return ptr;
}
template <typename memory_T,enable_IT<eq_CE(memory_T::location,DEVICE)> = 0>
cudaPitchedPtr __malloc__(const cudaExtent& extent,int device=-1) {
	cudaPitchedPtr ptr;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMalloc3D(&ptr,extent),MEMORY_ERROR);
	return ptr;
}
template <typename memory_T,enable_IT<eq_CE(memory_T::location,MANAGED)> = 0>
cudaPitchedPtr __malloc__(const cudaExtent& extent,int device=-1) {
	cudaPitchedPtr ptr;
	ptr.pitch=extent.width;
	ptr.xsize=extent.width;
	ptr.ysize=extent.height;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaMallocManaged((void **)&(ptr.ptr),extent.depth*extent.height*extent.width,memory_T::managed_F),MEMORY_ERROR);
	return ptr;
}
template <typename memory_T,typename T>
void __malloc__(T* &ptr,std::size_t &pitch,std::size_t xsize,std::size_t ysize,std::size_t zsize,int device=-1) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T,std::size_t>();
	cudaPitchedPtr auxPtr;
	cudaExtent extent=make_cudaExtent(xsize*sod,ysize,zsize);
	auxPtr=__malloc__<memory_T>(extent,device);
	ptr=auxPtr.ptr;
	pitch=auxPtr.pitch;
}
#endif
}
}
#endif
