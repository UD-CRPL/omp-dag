#ifndef __ARRAY_REDUCE_CALLERS_CUH__
#define __ARRAY_REDUCE_CALLERS_CUH__

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
#include "reduce-kernels.cuh"
#endif

#include "reduce-kernels.h"
#include "execution-policy.h"

namespace __core__ {
namespace __functional__ {
namespace __reduce__ {
namespace __array__ {
template <typename T=int> __host_device__ __optimize__ size_t reduceBufferSize(size_t size,bool syncpointsQ,size_t NIT,size_t blockdim) {
	size_t work=blockdim*NIT;
	size_t gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	size_t gridsize=(size+work-1)/work;
	return gridsize*gridsizeMultiplier*2;
}
template <bool syncpointsQ,int NIT,int blockdim> __host_device__ __optimize__ size_t reduceBufferSize(size_t size) {
	return reduceBufferSize(size,syncpointsQ,NIT,blockdim);
}
template <typename policy=ReduceArrayExecutionPolicy<>> __host_device__ __optimize__ size_t reduceBufferSize(size_t size) {
	return reduceBufferSize(size,policy::syncpointsQ,policy::NIT,policy::blockdim);
}

#if defined(CUDA_SUPPORT_COREQ)
template <typename RFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_gpu(RT *buffer,DT *arr,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data=(RT*)arr;
	if(size>minWork) {
		__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,(DT*)data,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,(DT*)data,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,typename IV,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_gpu(RT *buffer,DT *arr,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data=(RT*)arr;
	if(size>minWork) {
		__private__::__reduce_blocks_gkernel__<RFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,(DT*)data,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_blocks_gkernel__<RFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,(DT*)data,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_gpu(RT *buffer,DT *arr,IT size,RT init_value,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data=(RT*)arr;
	if(size>minWork) {
		__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,(DT*)data,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,(DT*)data,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}

template <typename RFT,typename AFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT *arr,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,typename AFT,typename IV,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT *arr,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,typename AFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT *arr,IT size,RT init_value,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}

template <typename RFT,typename AFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT1,DT2,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr1,arr2,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr1,arr2,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,typename AFT,typename IV,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT1,DT2,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
//	std::cerr<<"rapplyd:\t size:"<<size<<", grid size:"<<gridsize<<", work:"<<work<<", gridsizeMultiplier:"<<gridsizeMultiplier<<", gs wc:"<<gridsize*gridsizeMultiplier<<std::endl;
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr1,arr2,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr1,arr2,size);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,IV,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,IV,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
template <typename RFT,typename AFT,bool syncpointsQ,bool contiguousQ,int NIT,int blockdim,int minWork,bool unrolled=true,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void,enable_IT<!contains_type_CE<void,RT,DT1,DT2,IT>()> = 0> __optimize__
void reduce_apply_gpu(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,RT init_value,const cudaStream_t& stream=0) {
	constexpr IT work=blockdim*NIT;
	constexpr IT gridsizeMultiplier=syncpointsQ?1:(blockdim>>5);
	int gridsize=(size+work-1)/work;
	bool even=((__private__::__reduce_iterations_count__<syncpointsQ,NIT,blockdim,minWork,IT>(size)&1)==0);
	RT *pbuffer=even?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *sbuffer=(!even)?(buffer+gridsize*gridsizeMultiplier):buffer;
	RT *data;
	if(size>minWork) {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,arr1,arr2,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
	}
	else {
		__private__::__reduce_apply_blocks_gkernel__<RFT,AFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,arr1,arr2,size,init_value);
		check_cuda_error(KERNEL_LAUNCH_ERROR);
		return;
	}
	size=gridsize*gridsizeMultiplier;
	gridsize=(size+work-1)/work;
	data=pbuffer;
	pbuffer=sbuffer;
	sbuffer=data;
	while(gridsize>=1) {
		if(size>minWork) {
			__private__::__reduce_blocks_gkernel__<RFT,syncpointsQ,contiguousQ,blockdim,unrolled><<<gridsize,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
		}
		else {
			__private__::__reduce_blocks_gkernel__<RFT,true,contiguousQ,blockdim,unrolled><<<1,blockdim,0,stream>>>(pbuffer,data,size,init_value);
			check_cuda_error(KERNEL_LAUNCH_ERROR);
			return;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		data=pbuffer;
		pbuffer=sbuffer;
		sbuffer=data;
	}
}
#endif

template <typename RFT,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size) {
	return __private__::__reduce_ckernel__<RFT,threadnum>(result,arr,size);
}
template <typename RFT,typename IV,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size) {
	return __private__::__reduce_ckernel__<RFT,IV,threadnum>(result,arr,size);
}
template <typename RFT,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size,RT iv) {
	return __private__::__reduce_ckernel__<RFT,threadnum>(result,arr,size,iv);
}

template <typename RFT,typename AFT,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,threadnum>(result,arr,size);
}
template <typename RFT,typename AFT,typename IV,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,IV,threadnum>(result,arr,size);
}
template <typename RFT,typename AFT,int threadnum,typename RT=double,typename data_T=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T*) arr,IT size,const RT& iv) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,threadnum>(result,arr,size,iv);
}

template <typename RFT,typename AFT,int threadnum,typename RT=double,typename data_T1=double,typename data_T2=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T1*) arr1,CRESTRICT_Q(data_T1*) arr2,IT size) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,threadnum>(result,arr1,arr2,size);
}
template <typename RFT,typename AFT,typename IV,int threadnum,typename RT=double,typename data_T1=double,typename data_T2=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T1*) arr1,CRESTRICT_Q(data_T1*) arr2,IT size) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,IV,threadnum>(result,arr1,arr2,size);
}
template <typename RFT,typename AFT,int threadnum,typename RT=double,typename data_T1=double,typename data_T2=double,typename IT=int> __optimize__
RT reduce_apply_cpu(RT& result,CRESTRICT_Q(data_T1*) arr1,CRESTRICT_Q(data_T1*) arr2,IT size,const RT& iv) {
	return __private__::__reduce_apply_ckernel__<RFT,AFT,threadnum>(result,arr1,arr2,size,iv);
}

template <typename RFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_gpu<RFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_cpu<RFT,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_gpu<RFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_cpu<RFT,IV,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,RT init_value,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_gpu<RFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,init_value,stream);
#endif
	}
	else
		reduce_cpu<RFT,policy::threadnum>(*buffer,arr,size,init_value);
}
template <typename RFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_gpu<RFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_cpu<RFT,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_gpu<RFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_cpu<RFT,IV,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce(RT *buffer,DT *arr,IT size,RT init_value,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_gpu<RFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,init_value,stream);
#endif
	}
	else
		reduce_cpu<RFT,policy::threadnum>(*buffer,arr,size,init_value);
}

template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename AFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,IV,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,RT init_value,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,init_value,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr,size,init_value);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename AFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,IV,policy::threadnum>(*buffer,arr,size);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT=void,typename IT=void>
void reduce_apply(RT *buffer,DT *arr,IT size,RT init_value,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr,size,init_value,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr,size,init_value);
}

template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr1,arr2,size);
}
template <typename RFT,typename AFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,IV,policy::threadnum>(*buffer,arr1,arr2,size);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,RT init_value,int device,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		if(device>=0) {
			cuda_error(cudaSetDevice(device),API_ERROR);
		}
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,init_value,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr1,arr2,size,init_value);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr1,arr2,size);
}
template <typename RFT,typename AFT,typename IV,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,IV,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,IV,policy::threadnum>(*buffer,arr1,arr2,size);
}
template <typename RFT,typename AFT,DeviceType location,typename policy=ReduceArrayExecutionPolicy<>,typename RT=void,typename DT1=void,typename DT2=void,typename IT=void>
void reduce_apply(RT *buffer,DT1 *arr1,DT2 *arr2,IT size,RT init_value,const StreamType& stream=(StreamType)0) {
	if(location!=HOST) {
#if defined(CUDA_SUPPORT_COREQ)
		reduce_apply_gpu<RFT,AFT,policy::syncpointsQ,policy::contiguousQ,policy::NIT,policy::blockdim,policy::minWork>(buffer,arr1,arr2,size,init_value,stream);
#endif
	}
	else
		reduce_apply_cpu<RFT,AFT,policy::threadnum>(*buffer,arr1,arr2,size,init_value);
}
}
}
}
}
#endif
