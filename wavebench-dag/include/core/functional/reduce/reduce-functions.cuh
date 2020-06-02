#ifndef __REDUCE_FUNCTIONS_REDUCE_FUNCTIONAL_CORE_CUH__
#define __REDUCE_FUNCTIONS_REDUCE_FUNCTIONAL_CORE_CUH__

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../macros/macros.h"
#include "../../meta/meta.h"
#include "../../types/types.h"

namespace __core__ {
namespace __functional__ {
namespace __reduce__ {
//template <typename T,uint A> Vector<T,2,A> __shfl_down_sync(unsigned mask,const Vector<T,2,A>& v,int lane) {
//	Vector<T,2,A> result;
//	result[0]=__shfl_down_sync(mask,v[0],lane);
//	result[1]=__shfl_down_sync(mask,v[1],lane);
//	return result;
//}
//template <typename T,uint A> Vector<T,3,A> __shfl_down_sync(unsigned mask,const Vector<T,3,A>& v,int lane) {
//	Vector<T,3,A> result;
//	result[0]=__shfl_down_sync(mask,v[0],lane);
//	result[1]=__shfl_down_sync(mask,v[1],lane);
//	result[2]=__shfl_down_sync(mask,v[2],lane);
//	return result;
//}
template <typename fn_T,bool unrolled,typename T,enable_IT<unrolled==false> = 0> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_warp(T data) {
	for(int i=16;i>0;i=i>>1)
		data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,i),data);
	return data;
}
template <typename fn_T,bool unrolled,typename T,enable_IT<unrolled==true> = 0> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_warp(T data) {
	data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,16),data);
	data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,8),data);
	data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,4),data);
	data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,2),data);
	data=fn_T::fn(__shfl_down_sync(0xFFFFFFFF,data,1),data);
	return data;
}
template <typename fn_T,typename T> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_active(cooperative_groups::coalesced_group thread_group,T data) {
	T tmp;
	for(unsigned int i=1;i<thread_group.size();i=i<<1) {
		tmp=fn_T::fn(thread_group.shfl_down(data,i),data);
		if(i+thread_group.thread_rank()<thread_group.size())
			data=tmp;
	}
	return data;
}
template <bool unrolled,typename fn_T,typename T,enable_IT<unrolled==false> = 0> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_warp(T data,fn_T& fn) {
	for(int i=16;i>0;i=i>>1)
		data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,i),data);
	return data;
}
template <bool unrolled,typename fn_T,typename T,enable_IT<unrolled==true> = 0> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_warp(T data,fn_T& fn) {
	data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,16),data);
	data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,8),data);
	data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,4),data);
	data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,2),data);
	data=fn.reduce(__shfl_down_sync(0xFFFFFFFF,data,1),data);
	return data;
}
template <typename fn_T,typename T> __forceinline__ __forceflatten__ __optimize__ __device__ T reduce_active(cooperative_groups::coalesced_group thread_group,T data,fn_T& fn) {
	T tmp;
	for(unsigned int i=1;i<thread_group.size();i=i<<1) {
		tmp=fn.reduce(thread_group.shfl_down(data,i),data);
		if(i+thread_group.thread_rank()<thread_group.size())
			data=tmp;
	}
	return data;
}
template <typename reduce_FT,typename apply_FT,bool unrolled,typename T,typename...Args,enable_IT<unrolled==true> = 0> __forceflatten__ __optimize__ __device__ inline
T reduce_warp(T result,Args...args) {
	result=apply_FT::fn(result,args...);
	for(int i=16;i>0;i=i>>1)
		result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,i),result);
	return result;
}
template <typename reduce_FT,typename apply_FT,bool unrolled,typename T,typename...Args,enable_IT<unrolled!=true> = 0> __forceflatten__ __optimize__ __device__ inline
T reduce_warp(T result,Args...args) {
	result=apply_FT::fn(result,args...);
	result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,16),result);
	result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,8),result);
	result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,4),result);
	result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,2),result);
	result=reduce_FT::fn(__shfl_down_sync(0xFFFFFFFF,result,1),result);
	return result;
}
template <typename reduce_FT,typename apply_FT,typename T,typename...Args> __forceflatten__ __optimize__ __device__
T reduce_active(cooperative_groups::coalesced_group thread_group,T result,Args...args) {
	T tmp;
	result=apply_FT::fn(result,args...);
	for(unsigned int i=1;i<thread_group.size();i=i<<1) {
		tmp=reduce_FT::fn(thread_group.shfl_down(result,i),result);
		if(i+thread_group.thread_rank()<thread_group.size())
			result=tmp;
	}
	return result;
}
template <typename fn_T,int blockdim,bool unrolled=true,int warpCount=(blockdim>>5),typename T=void> __forceinline__ __forceflatten__ __optimize__ __device__
T reduce_block(T data) {
	__shared__ T tmp[warpCount];
	T result=reduce_warp<fn_T,unrolled>(data);
	if(threadIdx.x&31==0)
		tmp[threadIdx.x>>5]=result;
	if(threadIdx.x==0)
		for(int i=1;i<warpCount;++i)
			result=fn_T::fn(tmp[i],result);
	return result;
}
}
}
}
#endif
