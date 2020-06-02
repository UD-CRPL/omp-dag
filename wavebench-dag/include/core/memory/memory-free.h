#ifndef __MEMORY_FREE_MEMORY_CORE_H__
#define __MEMORY_FREE_MEMORY_CORE_H__

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

namespace __core__ {
namespace __memory__ {
//++++++++++++++++++++++++++++++++++++++++++++++++++ default versions ++++++++++++++++++++++++++++++++++++++++++++++++++
//*************************************************** unified memory ***************************************************
template <typename memory_T,typename T,enable_IT<eq_CE(memory_T::location,HOST)> = 0> __forceinline__ void __free__(T* ptr) {
	if(ptr==nullptr)
		return;
	if(!memory_T::is_pinned)
		std::free((void*)ptr);
#if defined(__CUDARUNTIMEQ__)
	else {
		cuda_error(cudaFreeHost(ptr),MEMORY_ERROR);
	}
#endif
}
#if defined(__CUDARUNTIMEQ__)
template <typename memory_T,typename T,enable_IT<!eq_CE(memory_T::location,HOST)> = 0> __forceinline__ void __free__(T* ptr) {
	if(ptr==nullptr)
		return;
	cuda_error(cudaFree(ptr),MEMORY_ERROR);
}
#endif
//************************************************* not unified memory *************************************************
template <typename memory_T,typename T,enable_IT<eq_CE(memory_T::location,HOST)> = 0> __forceinline__ void __free__(T* ptr,int device) {
	if(ptr==nullptr)
		return;
	if(!memory_T::is_pinned)
		std::free((void*)ptr);
#if defined(__CUDARUNTIMEQ__)
	else {
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		cuda_error(cudaFreeHost(ptr),MEMORY_ERROR);
	}
#endif
}
#if defined(__CUDARUNTIMEQ__)
template <typename memory_T,typename T,enable_IT<!eq_CE(memory_T::location,HOST)> = 0> __forceinline__ void __free__(T* ptr,int device) {
	if(ptr==nullptr)
		return;
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	cuda_error(cudaFree(ptr),MEMORY_ERROR);
}
#endif
}
}
#endif
