#ifndef __MEMORY_COPY_MEMORY_CORE_H__
#define __MEMORY_COPY_MEMORY_CORE_H__

#include <cstring>
#include <chrono>
#include <future>
#include <type_traits>

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
typedef cudaStream_t cudaStream_t; //Typedef to stop Nsight from complaining
typedef cudaMemoryAdvise cudaMemoryAdvise; //Typedef to stop Nsight from complaining
#endif
#include "../macros/compiler.h"

#include "../enum-definitions.h"
#include "../debug/debug.h"
#include "../meta/meta.h"
#include "util.h"

namespace __core__ {
namespace __memory__ {
//*************************************************** unified memory ***************************************************
//linear memory
#if defined(__CUDARUNTIMEQ__)
template <SyncBehaviorType sync_behavior=ASYNC,typename T=void> void __memcpy__(void *dst_ptr,void *src_ptr,std::size_t size,cudaStream_t stream=(cudaStream_t)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpyAsync(dst_ptr,src_ptr,size*sod,cudaMemcpyDefault,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy(dst_ptr,src_ptr,size*sod,cudaMemcpyDefault),MEMORY_ERROR);
	}
}

//2D memory
template <SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(void *dst_ptr,std::size_t dpitch,void *src_ptr,std::size_t spitch,std::size_t xsize,std::size_t ysize,cudaStream_t stream=(cudaStream_t)0) {
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpy2DAsync(dst_ptr,dpitch,src_ptr,spitch,xsize,ysize,cudaMemcpyDefault,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy2D(dst_ptr,dpitch,src_ptr,spitch,xsize,ysize,cudaMemcpyDefault),MEMORY_ERROR);
	}
}

//3D memory
template <SyncBehaviorType sync_behavior=ASYNC> void __memcpy__(cudaMemcpy3DParms &params,cudaStream_t stream=(cudaStream_t)0) {
	params.kind=cudaMemcpyDefault;
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpy3DAsync(&params,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy3D(&params),MEMORY_ERROR);
	}
}
#endif

//************************************************* not unified memory *************************************************
//linear memory
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=SYNC,typename T=void,enable_IT<(__memory_private__::transfer_type<memory_ST,memory_DT>()==CPU_CPU)&&(sync_behavior==SYNC)> = 0>
void __memcpy__(T *dst_ptr,T *src_ptr,std::size_t size,int device=0,StreamType stream=(StreamType)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	memcpy(reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod);
}
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=SYNC,typename T=void,enable_IT<(__memory_private__::transfer_type<memory_ST,memory_DT>()==CPU_CPU)&&(sync_behavior==ASYNC)> = 0>
std::future<void*> __memcpy__(T *dst_ptr,T *src_ptr,std::size_t size,int device=0,StreamType stream=(StreamType)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	return std::async(std::launch::async,memcpy,reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod);
}

#if defined(__CUDARUNTIMEQ__)
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<neq_CE(__memory_private__::transfer_type<memory_ST,memory_DT>(),CPU_CPU)> = 0>
void __memcpy__(T *dst_ptr,T *src_ptr,std::size_t size,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	constexpr cudaMemcpyKind mcpykind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod,mcpykind,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy(reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod,mcpykind),MEMORY_ERROR);
	}
}
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<(__memory_private__::transfer_type<memory_ST,memory_DT>()==GPU_GPU)> = 0>
void __memcpy__(T *dst_ptr,T *src_ptr,std::size_t size,int dst_device,int src_device,cudaStream_t stream=(cudaStream_t)0) {
	constexpr std::size_t sod=__memory_private__::__sizeof__<T>();
	if(dst_device>=0) {
		cuda_error(cudaSetDevice(dst_device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		if(dst_device!=src_device) {
			cuda_error(cudaMemcpyPeerAsync(reinterpret_cast<void*>(dst_ptr),dst_device,reinterpret_cast<void*>(src_ptr),src_device,size*sod,stream),MEMORY_ERROR);
		}
		else {
			cuda_error(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod,cudaMemcpyDeviceToDevice,stream),MEMORY_ERROR);
		}
	}
	else {
		if(dst_device!=src_device) {
			cuda_error(cudaMemcpyPeer(reinterpret_cast<void*>(dst_ptr),dst_device,reinterpret_cast<void*>(src_ptr),src_device,size*sod),MEMORY_ERROR);
		}
		else {
			cuda_error(cudaMemcpy(reinterpret_cast<void*>(dst_ptr),reinterpret_cast<void*>(src_ptr),size*sod,cudaMemcpyDeviceToDevice),MEMORY_ERROR);
		}
	}
}
#endif

//2D memory

#if defined(__CUDARUNTIMEQ__)
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC> void __memcpy__(void *dst_ptr,std::size_t dpitch,void *src_ptr,std::size_t spitch,std::size_t xsize,std::size_t ysize,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	constexpr cudaMemcpyKind mcpykind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpy2DAsync(dst_ptr,dpitch,src_ptr,spitch,xsize,ysize,mcpykind,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy2D(dst_ptr,dpitch,src_ptr,spitch,xsize,ysize,mcpykind),MEMORY_ERROR);
	}
}
#endif

//3D memory
#if defined(__CUDARUNTIMEQ__)
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaMemcpy3DParms &params,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	if(device>=0) {
		cuda_error(cudaSetDevice(device),API_ERROR);
	}
	params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpy3DAsync(&params,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy3D(&params),MEMORY_ERROR);
	}
}
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaMemcpy3DParms &params,int dst_device,int src_device,cudaStream_t stream=(cudaStream_t)0) {
	if(dst_device>=0) {
		cuda_error(cudaSetDevice(dst_device),API_ERROR);
	}
	if((dst_device!=src_device)&&(__memory_private__::transfer_type<memory_ST,memory_DT>()==GPU_GPU)) {
		cudaMemcpy3DPeerParms paramsp={0};
		paramsp.dstPtr=params.dstPtr;
		paramsp.dstPos=params.dstPos;
		paramsp.dstDevice=dst_device;
		paramsp.srcPtr=params.srcPtr;
		paramsp.srcPos=params.srcPos;
		paramsp.srcDevice=src_device;
		paramsp.extent=params.extent;
		if(sync_behavior==ASYNC) {
			cuda_error(cudaMemcpy3DPeerAsync(&paramsp,stream),MEMORY_ERROR);
		}
		else {
			cuda_error(cudaMemcpy3DPeer(&paramsp),MEMORY_ERROR);
		}
	}
	else {
		params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
		if(sync_behavior==ASYNC) {
			cuda_error(cudaMemcpy3DAsync(&params,stream),MEMORY_ERROR);
		}
		else {
			cuda_error(cudaMemcpy3D(&params),MEMORY_ERROR);
		}
	}
}
template <SyncBehaviorType sync_behavior=ASYNC> void __memcpy__(cudaMemcpy3DPeerParms &params,cudaStream_t stream=(cudaStream_t)0) {
	if(params.dstDevice>=0) {
		cuda_error(cudaSetDevice(params.dstDevice),API_ERROR);
	}
	if(sync_behavior==ASYNC) {
		cuda_error(cudaMemcpy3DPeerAsync(&params,stream),MEMORY_ERROR);
	}
	else {
		cuda_error(cudaMemcpy3DPeer(&params),MEMORY_ERROR);
	}
}

template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaPitchedPtr &dst_ptr,const cudaPitchedPtr &src_ptr,const cudaExtent &extent,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	cudaMemcpy3DParms params={0};
	params.dstPtr=dst_ptr;
	params.srcPtr=src_ptr;
	params.extent=extent;
	params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	__memcpy__<memory_DT,memory_ST,sync_behavior>(params,device,stream);
}
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaPitchedPtr &dst_ptr,const cudaPitchedPtr &src_ptr,const cudaExtent &extent,const cudaPos &dst_pos,const cudaPos &src_pos,int device=-1,cudaStream_t stream=(cudaStream_t)0) {
	cudaMemcpy3DParms params={0};
	params.dstPtr=dst_ptr;
	params.dstPos=dst_pos;
	params.srcPtr=src_ptr;
	params.srcPos=src_pos;
	params.extent=extent;
	params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	__memcpy__<memory_DT,memory_ST,sync_behavior>(params,device,stream);
}

template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaPitchedPtr &dst_ptr,const cudaPitchedPtr &src_ptr,const cudaExtent &extent,int dst_dev,int src_dev,cudaStream_t stream=(cudaStream_t)0) {
	cudaMemcpy3DParms params={0};
	params.dstPtr=dst_ptr;
	params.srcPtr=src_ptr;
	params.extent=extent;
	params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	__memcpy__<memory_DT,memory_ST,sync_behavior>(params,dst_dev,src_dev,stream);
}
template <typename memory_DT,typename memory_ST,SyncBehaviorType sync_behavior=ASYNC>
void __memcpy__(cudaPitchedPtr &dst_ptr,const cudaPitchedPtr &src_ptr,const cudaExtent &extent,const cudaPos &dst_pos,const cudaPos &src_pos,int dst_dev,int src_dev,cudaStream_t stream=(cudaStream_t)0) {
	cudaMemcpy3DParms params={0};
	params.dstPtr=dst_ptr;
	params.dstPos=dst_pos;
	params.srcPtr=src_ptr;
	params.srcPos=src_pos;
	params.extent=extent;
	params.kind=__memory_private__::memcpykind<memory_ST,memory_DT>();
	__memcpy__<memory_DT,memory_ST,sync_behavior>(params,dst_dev,src_dev,stream);
}
#endif
}
}
#endif
