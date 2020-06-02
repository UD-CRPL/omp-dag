#ifndef __ARRAY_APPLY_CALLER_FUNCTIONAL_CORE_H__
#define __ARRAY_APPLY_CALLER_FUNCTIONAL_CORE_H__
#include <type_traits>

#include "../../../macros/definitions.h"
#if defined(CUDA_SUPPORT_COREQ)
#include <cuda_runtime.h>
#endif
#include "../../../macros/compiler.h"
#include "../../../macros/functions.h"

#include "../../../debug/debug.h"
#include "../../../meta/meta.h"
#include "../../../types/types.h"
#if defined(CUDA_SUPPORT_COREQ)
#include "apply-kernels.cuh"
#endif
#include "apply-kernels.h"
#include "execution-policy.h"

namespace __core__ {
namespace __functional__ {
namespace __apply__ {
namespace __array__ {
#if defined(CUDA_SUPPORT_COREQ)
template <typename fn_T,bool sync_points=false,ReadMode read_mode=read_only,int NIT=32,int blockdim=256,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_gpu(T* arr,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_gkernel__<fn_T,sync_points,read_mode,blockdim,T,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode=read_only,int NIT=32,int blockdim=256,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_gpu(T* arr_dst,U* arr_src,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_gkernel__<fn_T,sync_points,read_mode,blockdim,T,U,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr_src,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode1=read_only,ReadMode read_mode2=read_only,int NIT=32,int blockdim=256,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_gpu(T* arr_dst,V* arr1,U* arr2,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_gkernel__<fn_T,sync_points,read_mode1,read_mode2,blockdim,T,V,U,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr1,arr2,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode1=read_only,ReadMode read_mode2=read_only,ReadMode read_mode3=read_only,int NIT=32,int blockdim=256,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_gpu(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_gkernel__<fn_T,sync_points,read_mode1,read_mode2,read_mode3,blockdim,T,V,U,W,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr1,arr2,arr3,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode=read_only,int NIT=32,int blockdim=256,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_gpu(T* arr,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_indexed_gkernel__<fn_T,sync_points,read_mode,blockdim,T,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode=read_only,int NIT=32,int blockdim=256,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_gpu(T* arr_dst,U* arr_src,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_indexed_gkernel__<fn_T,sync_points,read_mode,blockdim,T,U,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr_src,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode1=read_only,ReadMode read_mode2=read_only,int NIT=32,int blockdim=256,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_gpu(T* arr_dst,V* arr1,U* arr2,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_indexed_gkernel__<fn_T,sync_points,read_mode1,read_mode2,blockdim,T,V,U,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr1,arr2,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points=false,ReadMode read_mode1=read_only,ReadMode read_mode2=read_only,ReadMode read_mode3=read_only,int NIT=32,int blockdim=256,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_gpu(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,const cudaStream_t& stream=0,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_indexed_gkernel__<fn_T,sync_points,read_mode1,read_mode2,read_mode3,blockdim,T,V,U,W,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(arr_dst,arr1,arr2,arr3,size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}

template <typename fn_T,bool sync_points,int NIT=32,int blockdim,typename IT,typename... Args> __optimize__ __forceflatten__
void apply_meta_gpu(IT size,cudaStream_t stream,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_meta_gkernel__<fn_T,sync_points,false,blockdim,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,0,stream>>>(size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
template <typename fn_T,bool sync_points,int NIT=32,int blockdim,typename IT,typename... Args> __optimize__ __forceflatten__
void apply_meta_gpu(IT size,IT shared_mem_size,cudaStream_t stream,Args... args) {
	constexpr IT work=blockdim*NIT;
	int gridsize=(size+work-1)/work;
	__private__::__apply_function_meta_gkernel__<fn_T,sync_points,true,blockdim,IT,log2_CE(blockdim),0,Args...><<<gridsize,blockdim,shared_mem_size,stream>>>(size,args...);
	check_cuda_error(KERNEL_LAUNCH_ERROR);
}
#endif

template <typename fn_T,int threadnum=8,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_cpu(T* arr,IT size,Args... args) {
	__private__::__apply_function_ckernel__<fn_T,threadnum>(arr,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_cpu(T* arr_dst,U* arr_src,IT size,Args... args) {
	__private__::__apply_function_ckernel__<fn_T,threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_cpu(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	__private__::__apply_function_ckernel__<fn_T,threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_cpu(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	__private__::__apply_function_ckernel__<fn_T,threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_cpu(T* arr,IT size,Args... args) {
	__private__::__apply_function_indexed_ckernel__<fn_T,threadnum>(arr,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_cpu(T* arr_dst,U* arr_src,IT size,Args... args) {
	__private__::__apply_function_indexed_ckernel__<fn_T,threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_cpu(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	__private__::__apply_function_indexed_ckernel__<fn_T,threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,int threadnum=8,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed_cpu(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	__private__::__apply_function_indexed_ckernel__<fn_T,threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}

template <typename fn_T,int threadnum=8,typename IT=int,typename... Args> __optimize__ __forceflatten__
void apply_meta_cpu(IT size,Args... args) {
	__private__::__apply_function_meta_ckernel__<fn_T,threadnum,IT,Args...>(size,args...);
}
template <typename fn_T,int threadnum=8,typename IT=int,typename... Args> __optimize__ __forceflatten__
void apply_meta_cpu(IT size,IT shared_mem_size,Args... args) {
	__private__::__apply_function_meta_ckernel__<fn_T,threadnum,IT,Args...>(size,shared_mem_size,args...);
}

template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,U* arr_src,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr,IT size,const StreamType& stream,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,U* arr_src,IT size,const StreamType& stream,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,IT size,const StreamType& stream,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,const StreamType& stream,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,stream,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,0,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,U* arr_src,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,0,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,0,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,0,args...);
#endif
	}
	else
		apply_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}

template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,U* arr_src,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr,IT size,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,U* arr_src,IT size,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,IT size,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,stream,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr,size,0,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,U* arr_src,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode,policy::NIT,policy::blockdim>(arr_dst,arr_src,size,0,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr_src,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,size,0,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,size,args...);
}
template <typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename T=void,typename V=void,typename U=void,typename W=void,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_indexed(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_indexed_gpu<fn_T,policy::sync_points,policy::read_mode1,policy::read_mode2,policy::read_mode3,policy::NIT,policy::blockdim>(arr_dst,arr1,arr2,arr3,size,0,args...);
#endif
	}
	else
		apply_indexed_cpu<fn_T,policy::threadnum>(arr_dst,arr1,arr2,arr3,size,args...);
}

template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_meta(IT size,int device,const StreamType& stream,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_meta_gpu<fn_T,policy::sync_points,policy::NIT,policy::blockdim>(size,stream,args...);
#endif
	}
	else
		apply_meta_cpu<fn_T,policy::threadnum>(size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_meta(IT size,IT shared_mem_size,int device,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		apply_meta_gpu<fn_T,policy::sync_points,policy::NIT,policy::blockdim>(size,shared_mem_size,stream,args...);
#endif
	}
	else
		apply_meta_cpu<fn_T,policy::threadnum>(size,shared_mem_size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_meta(IT size,const StreamType& stream=(StreamType)0,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_meta_gpu<fn_T,policy::sync_points,policy::NIT,policy::blockdim>(size,stream,args...);
#endif
	}
	else
		apply_meta_cpu<fn_T,policy::threadnum>(size,args...);
}
template<typename fn_T,DeviceType location,typename policy=ApplyArrayExecutionPolicy<>,typename IT=int,typename... Args> __forceinline__ __optimize__
void apply_meta(IT size,Args... args) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		apply_meta_gpu<fn_T,policy::sync_points,policy::NIT,policy::blockdim>(size,0,args...);
#endif
	}
	else
		apply_meta_cpu<fn_T,policy::threadnum>(size,args...);
}
}
}
}
}
#endif
