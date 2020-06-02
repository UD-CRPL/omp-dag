#ifndef __REDUCE_KERNELS_ARRAY_FUNCTIONAL_CORE_CUH__
#define __REDUCE_KERNELS_ARRAY_FUNCTIONAL_CORE_CUH__

#include <type_traits>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../../macros/macros.h"
#include "../../../types/types.h"
#include "../../../meta/meta.h"
#include "../reduce-functions.cuh"

namespace __core__ {
namespace __functional__ {
namespace __reduce__ {
namespace __array__ {
namespace __private__ {
template <bool syncpointQ,int NIT,int blockdim,int minWork,typename IT=size_t,int work=blockdim*NIT,IT gridsizeMultiplier=syncpointQ?1:(blockdim>>5)> __host_device__ __optimize__
int __reduce_iterations_count__(IT size) {
	int gridsize=(size+work-1)/work;
	int it=0;
	while(gridsize>=1) {
		if(size<=minWork) {
			++it;
			break;
		}
		size=gridsize*gridsizeMultiplier;
		gridsize=(size+work-1)/work;
		++it;
	}
	return it;
}

template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			result_local=read_memory<read_only>(arr+idx);
			result_local=reduce_warp<fn_T,unrolled>(result_local);
		}
		else {
			if(idx<size) {
				result_local=read_memory<read_only>(arr+idx);
				result_local=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),result_local);
			}
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data=read_memory<read_only>(arr+idx);
				data=reduce_warp<fn_T,unrolled>(data);
			}
			else {
				if(idx<size) {
					data=read_memory<read_only>(arr+idx);
					data=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=fn_T::fn(data,result_local);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			data=read_memory<read_only>(arr+idx);
			data=reduce_warp<fn_T,unrolled>(data);
		}
		else {
			if(idx<size) {
				data=read_memory<read_only>(arr+idx);
				data=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),data);
			}
		}
		result_local=data;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data=read_memory<read_only>(arr+idx);
				data=reduce_warp<fn_T,unrolled>(data);
			}
			else {
				if(idx<size) {
					data=read_memory<read_only>(arr+idx);
					data=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=fn_T::fn(data,result_local);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	uint thread_executed=0;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1U;
			data=read_memory<read_only>(arr+idx);
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size)
				data=fn_T::fn(read_memory<read_only>(arr+idx),data);
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<fn_T,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),data);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	uint thread_executed=0;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1;
			data=read_memory<read_only>(arr+idx);
		}
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size)
				data=fn_T::fn(read_memory<read_only>(arr+idx),data);
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<fn_T,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<fn_T>(cooperative_groups::coalesced_threads(),data);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}

template <typename fn_T,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data,result_local=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=read_memory<read_only>(arr+idx);
		else
			data=IV::value;
		data=reduce_warp<fn_T,unrolled>(data);
		result_local=fn_T::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<fn_T,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename fn_T,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data,r=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=read_memory<read_only>(arr+idx);
		else
			data=IV::value;
		data=reduce_warp<fn_T,unrolled>(data);
		r=fn_T::fn(data,r);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=r;
}
template <typename fn_T,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data=IV::value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=fn_T::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<fn_T,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<fn_T,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename fn_T,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data=IV::value,result_local=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=fn_T::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<fn_T,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}

template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data,result_local=init_value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=read_memory<read_only>(arr+idx);
		else
			data=init_value;
		data=reduce_warp<fn_T,unrolled>(data);
		result_local=fn_T::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<fn_T,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data,result_local=init_value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=read_memory<read_only>(arr+idx);
		else
			data=init_value;
		data=reduce_warp<fn_T,unrolled>(data);
		result_local=fn_T::fn(data,result_local);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data=init_value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=fn_T::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<fn_T,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<fn_T,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename fn_T,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data=init_value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=fn_T::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<fn_T,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}

template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			result_local=AFT::fn(read_memory<read_only>(arr+idx));
			result_local=reduce_warp<RFT,unrolled>(result_local);
		}
		else {
			if(idx<size) {
				result_local=AFT::fn(read_memory<read_only>(arr+idx));
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),result_local);
			}
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data=AFT::fn(read_memory<read_only>(arr+idx));
				data=reduce_warp<RFT,unrolled>(data);
			}
			else {
				if(idx<size) {
					data=AFT::fn(read_memory<read_only>(arr+idx));
					data=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=RFT::fn(data,result_local);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			data=AFT::fn(read_memory<read_only>(arr+idx));
			data=reduce_warp<RFT,unrolled>(data);
		}
		else {
			if(idx<size) {
				data=AFT::fn(read_memory<read_only>(arr+idx));
				data=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
			}
		}
		result_local=data;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data=AFT::fn(read_memory<read_only>(arr+idx));
				data=reduce_warp<RFT,unrolled>(data);
			}
			else {
				if(idx<size) {
					data=AFT::fn(read_memory<read_only>(arr+idx));
					data=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=RFT::fn(data,result_local);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	uint thread_executed=0;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1U;
			data=AFT::fn(read_memory<read_only>(arr+idx));
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size)
				data=AFT::fn(read_memory<read_only>(arr+idx),data);
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<RFT,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	uint thread_executed=0;
	RT data,result_local;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1;
			data=AFT::fn(read_memory<read_only>(arr+idx));
		}
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size)
				data=AFT::fn(read_memory<read_only>(arr+idx),data);
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<RFT,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}

template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data,result_local=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx));
		else
			data=IV::value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data,r=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx));
		else
			data=IV::value;
		data=reduce_warp<RFT,unrolled>(data);
		r=RFT::fn(data,r);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=r;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data=IV::value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data=IV::value,result_local=IV::value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}

template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data,result_local=init_value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx));
		else
			data=init_value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data,result_local=init_value;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx));
		else
			data=init_value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT data=init_value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,CRESTRICT_Q(DT*) arr,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT data=init_value,result_local;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size)
			data=AFT::fn(read_memory<read_only>(arr+idx),data);
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}

template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT result_local,data;
	DT1 data1;
	DT2 data2;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			result_local=AFT::fn(data1,data2);
			result_local=reduce_warp<RFT,unrolled>(result_local);
		}
		else {
			if(idx<size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				result_local=AFT::fn(data1,data2);
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),result_local);
			}
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				data=AFT::fn(data1,data2);
				data=reduce_warp<RFT,unrolled>(data);
			}
			else {
				if(idx<size) {
					data1=read_memory<read_only>(arr1+idx);
					data2=read_memory<read_only>(arr2+idx);
					data=AFT::fn(data1,data2);
					data=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=RFT::fn(data,result_local);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT result_local,data;
	DT1 data1;
	DT2 data2;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		IT nwbindx=wbindx+32;
		if(nwbindx<=size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			result_local=AFT::fn(data1,data2);
			result_local=reduce_warp<RFT,unrolled>(result_local);
		}
		else {
			if(idx<size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				result_local=AFT::fn(data1,data2);
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),result_local);
			}
		}
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			nwbindx=wbindx+32;
			if(nwbindx<=size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				data=AFT::fn(data1,data2);
				data=reduce_warp<RFT,unrolled>(data);
			}
			else {
				if(idx<size) {
					data1=read_memory<read_only>(arr1+idx);
					data2=read_memory<read_only>(arr2+idx);
					data=AFT::fn(data1,data2);
					data=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
				}
			}
			result_local=RFT::fn(data,result_local);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	__shared__ RT tmp_arr[warpCount];
	__shared__ bool executed[warpCount];
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	uint thread_executed=0;
	RT result_local,data;
	DT1 data1;
	DT2 data2;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1;
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		if(laneid==0)
			executed[warpid]=true;
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				data=AFT::fn(data1,data2,data);
			}
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<RFT,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
		}
		if(laneid==0)
			tmp_arr[warpid]=result_local;
	}
	else {
		if(laneid==0)
			executed[warpid]=false;
	}
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount) {
		if(executed[laneid]) {
			result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),tmp_arr[laneid]);
			if(laneid==0)
				*(result+blockIdx.x)=result_local;
		}
	}
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	RT result_local,data;
	uint thread_executed=0;
	DT1 data1;
	DT2 data2;
	if(wbindx<size) {
		IT idx=wbindx+laneid;
		if(idx<size) {
			thread_executed=1U;
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		wbindx+=step;
		for(;wbindx<size;wbindx+=step) {
			idx=wbindx+laneid;
			if(idx<size) {
				data1=read_memory<read_only>(arr1+idx);
				data2=read_memory<read_only>(arr2+idx);
				data=AFT::fn(data1,data2,data);
			}
		}
		idx=1<<(laneid);
		wbindx=__ballot_sync(0xFFFFFFFF,thread_executed);
		if(wbindx==0xFFFFFFFF)
			result_local=reduce_warp<RFT,unrolled>(data);
		else {
			if((wbindx&idx)==idx)
				result_local=reduce_active<RFT>(cooperative_groups::coalesced_threads(),data);
		}
		idx=warpid+blockIdx.x*warpCount;
		if(laneid==0)
			*(result+idx)=result_local;
		return;
	}
	else
		return;
}

template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT result_local=IV::value,data;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		else
			data=IV::value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT result_local=IV::value,data;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		else
			data=IV::value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT result_local,data=IV::value;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2,data);
		}
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=IV::value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,typename IV,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT result_local,data=IV::value;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2,data);
		}
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}

template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT result_local=init_value,data;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		else
			data=init_value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==true)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT result_local=init_value,data;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2);
		}
		else
			data=init_value;
		data=reduce_warp<RFT,unrolled>(data);
		result_local=RFT::fn(data,result_local);
	}
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==true)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size,const RT init_value) {
	cooperative_groups::thread_block thread_block=cooperative_groups::this_thread_block();
	__shared__ RT tmp_arr[warpCount];
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	const IT bindx=blockIdx.x<<bdimLOG2;
	IT wbindx=(warpid<<5)+bindx;
	IT idx;
	RT result_local,data=init_value;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2,data);
		}
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	if(laneid==0)
		tmp_arr[warpid]=result_local;
	thread_block.sync();
	if(warpid>0)
		return;
	if(laneid<warpCount)
		result_local=tmp_arr[laneid];
	else
		result_local=init_value;
	result_local=reduce_warp<RFT,unrolled>(result_local);
	if(laneid==0)
		*(result+blockIdx.x)=result_local;
}
template <typename RFT,typename AFT,bool syncpointQ,bool contiguousQ,int blockdim,bool unrolled,typename RT,typename DT1,typename DT2,typename IT,IT bdimLOG2=log2_CE(blockdim),IT warpCount=(blockdim>>5),enable_IT<(syncpointQ==false)&&(contiguousQ==false)> = 0> __forceflatten__ __optimize__ __global__
void __reduce_apply_blocks_gkernel__(RT *result,DT1 *arr1,DT2 *arr2,IT size,const RT init_value) {
	const IT warpid=threadIdx.x>>5;
	const IT step=gridDim.x<<bdimLOG2;
	const IT laneid=threadIdx.x&31;
	IT wbindx=(warpid<<5)+(blockIdx.x<<bdimLOG2);
	IT idx;
	RT result_local,data=init_value;
	DT1 data1;
	DT2 data2;
	for(;wbindx<size;wbindx+=step) {
		idx=wbindx+laneid;
		if(idx<size) {
			data1=read_memory<read_only>(arr1+idx);
			data2=read_memory<read_only>(arr2+idx);
			data=AFT::fn(data1,data2,data);
		}
	}
	result_local=reduce_warp<RFT,unrolled>(data);
	idx=warpid+blockIdx.x*warpCount;
	if(laneid==0)
		*(result+idx)=result_local;
}
}
}
}
}
}
#endif
