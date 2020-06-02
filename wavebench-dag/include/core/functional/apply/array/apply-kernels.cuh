#ifndef __ARRAY_APPLY_KERNELS_FUNCTIONAL_CORE_CUH__
#define __ARRAY_APPLY_KERNELS_FUNCTIONAL_CORE_CUH__

#include <type_traits>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../../macros/macros.h"
#include "../../../meta/meta.h"
#include "../../../types/types.h"

namespace __core__ {
namespace __functional__ {
namespace __apply__ {
namespace __array__ {
namespace __private__ {
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		T data=read_memory<read_mode>(arr+idx);
		arr[idx]=fn_T::fn(data,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			T data=read_memory<read_mode>(arr+idx);
			arr[idx]=fn_T::fn(data,args...);
		}
	}
	if(bidx<size) {
		T data;
		if(idx<size)
			data=read_memory<read_mode>(arr+idx);
		data=fn_T::fn(data,args...);
		if(idx<size)
			arr[idx]=data;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__  __global__
void __apply_function_gkernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		U data=read_memory<read_mode>(arr_src+idx);
		arr_dst[idx]=fn_T::fn(data,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__  __global__
void __apply_function_gkernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			U data_src=read_memory<read_mode>(arr_src+idx);
			arr_dst[idx]=fn_T::fn(data_src,args...);
		}
	}
	if(bidx<size) {
		U data_src;
		T data_dst;
		if(idx<size)
			data_src=read_memory<read_mode>(arr_src+idx);
		data_dst=fn_T::fn(data_src,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,int blockdim,typename T,typename V,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		V data1=read_memory<read_mode1>(arr1+idx);
		U data2=read_memory<read_mode2>(arr2+idx);
		arr_dst[idx]=fn_T::fn(data1,data2,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,int blockdim,typename T,typename V,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			V data1=read_memory<read_mode1>(arr1+idx);
			U data2=read_memory<read_mode2>(arr2+idx);
			arr_dst[idx]=fn_T::fn(data1,data2,args...);
		}
	}
	if(bidx<size) {
		V data1;
		U data2;
		T data_dst;
		if(idx<size) {
			data1=read_memory<read_mode1>(arr1+idx);
			data2=read_memory<read_mode2>(arr2+idx);
		}
		data_dst=fn_T::fn(data1,data2,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,ReadMode read_mode3,int blockdim,typename T,typename V,typename U,typename W,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		V data1=read_memory<read_mode1>(arr1+idx);
		U data2=read_memory<read_mode2>(arr2+idx);
		W data3=read_memory<read_mode3>(arr3+idx);
		arr_dst[idx]=fn_T::fn(data1,data2,data3,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,ReadMode read_mode3,int blockdim,typename T,typename V,typename U,typename W,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_gkernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			V data1=read_memory<read_mode1>(arr1+idx);
			U data2=read_memory<read_mode2>(arr2+idx);
			W data3=read_memory<read_mode3>(arr3+idx);
			arr_dst[idx]=fn_T::fn(data1,data2,data3,args...);
		}
	}
	if(bidx<size) {
		V data1;
		U data2;
		W data3;
		T data_dst;
		if(idx<size) {
			data1=read_memory<read_mode1>(arr1+idx);
			data2=read_memory<read_mode2>(arr2+idx);
			data3=read_memory<read_mode3>(arr3+idx);
		}
		data_dst=fn_T::fn(data1,data2,data3,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}

template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		T data=read_memory<read_mode>(arr+idx);
		arr[idx]=fn_T::fn(idx,data,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			T data=read_memory<read_mode>(arr+idx);
			arr[idx]=fn_T::fn(idx,data,args...);
		}
	}
	if(bidx<size) {
		T data;
		if(idx<size)
			data=read_memory<read_mode>(arr+idx);
		data=fn_T::fn(idx,data,args...);
		if(idx<size)
			arr[idx]=data;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__  __global__
void __apply_function_indexed_gkernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		U data=read_memory<read_mode>(arr_src+idx);
		arr_dst[idx]=fn_T::fn(idx,data,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode,int blockdim,typename T,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__  __global__
void __apply_function_indexed_gkernel__(T* arr_dst,U* arr_src,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			U data_src=read_memory<read_mode>(arr_src+idx);
			arr_dst[idx]=fn_T::fn(idx,data_src,args...);
		}
	}
	if(bidx<size) {
		U data_src;
		T data_dst;
		if(idx<size)
			data_src=read_memory<read_mode>(arr_src+idx);
		data_dst=fn_T::fn(idx,data_src,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,int blockdim,typename T,typename V,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		V data1=read_memory<read_mode1>(arr1+idx);
		U data2=read_memory<read_mode2>(arr2+idx);
		arr_dst[idx]=fn_T::fn(idx,data1,data2,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,int blockdim,typename T,typename V,typename U,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr_dst,V* arr1,U* arr2,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			V data1=read_memory<read_mode1>(arr1+idx);
			U data2=read_memory<read_mode2>(arr2+idx);
			arr_dst[idx]=fn_T::fn(idx,data1,data2,args...);
		}
	}
	if(bidx<size) {
		V data1;
		U data2;
		T data_dst;
		if(idx<size) {
			data1=read_memory<read_mode1>(arr1+idx);
			data2=read_memory<read_mode2>(arr2+idx);
		}
		data_dst=fn_T::fn(idx,data1,data2,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,ReadMode read_mode3,int blockdim,typename T,typename V,typename U,typename W,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==false> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step) {
		V data1=read_memory<read_mode1>(arr1+idx);
		U data2=read_memory<read_mode2>(arr2+idx);
		W data3=read_memory<read_mode3>(arr3+idx);
		arr_dst[idx]=fn_T::fn(idx,data1,data2,data3,args...);
	}
}
template <typename fn_T,bool sync_points,ReadMode read_mode1,ReadMode read_mode2,ReadMode read_mode3,int blockdim,typename T,typename V,typename U,typename W,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<sync_points==true> = 0,typename... Args> __optimize__ __global__
void __apply_function_indexed_gkernel__(T* arr_dst,V* arr1,U* arr2,W* arr3,IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step) {
			V data1=read_memory<read_mode1>(arr1+idx);
			U data2=read_memory<read_mode2>(arr2+idx);
			W data3=read_memory<read_mode3>(arr3+idx);
			arr_dst[idx]=fn_T::fn(idx,data1,data2,data3,args...);
		}
	}
	if(bidx<size) {
		V data1;
		U data2;
		W data3;
		T data_dst;
		if(idx<size) {
			data1=read_memory<read_mode1>(arr1+idx);
			data2=read_memory<read_mode2>(arr2+idx);
			data3=read_memory<read_mode3>(arr3+idx);
		}
		data_dst=fn_T::fn(idx,data1,data2,data3,args...);
		if(idx<size)
			arr_dst[idx]=data_dst;
	}
}

template <typename fn_T,bool sync_points,bool shared_memory,int blockdim,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<(shared_memory==false)&&(sync_points==false)> = 0,typename... Args> __optimize__ __forceflatten__ __global__
void __apply_function_meta_gkernel__(IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step)
		fn_T::fn(idx,size,args...);
}
template <typename fn_T,bool sync_points,bool shared_memory,int blockdim,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<(shared_memory==true)&&(sync_points==false)> = 0,typename... Args> __optimize__ __forceflatten__ __global__
void __apply_function_meta_gkernel__(IT size,Args... args) {
	extern __shared__ uchar shared[];
	const IT step=gridDim.x<<BDIML2;
	IT idx=(blockIdx.x<<BDIML2)+threadIdx.x;
	for(;idx<size;idx+=step)
		fn_T::fn(idx,size,shared,args...);
}
template <typename fn_T,bool sync_points,bool shared_memory,int blockdim,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<(shared_memory==false)&&(sync_points==true)> = 0,typename... Args> __optimize__ __forceflatten__ __global__
void __apply_function_meta_gkernel__(IT size,Args... args) {
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step)
			fn_T::fn(idx,size,args...);
	}
	if(bidx<size)
		fn_T::fn(idx,size,args...);
}
template <typename fn_T,bool sync_points,bool shared_memory,int blockdim,typename IT,IT BDIML2=log2_CE(blockdim),enable_IT<(shared_memory==true)&&(sync_points==true)> = 0,typename... Args> __optimize__ __forceflatten__ __global__
void __apply_function_meta_gkernel__(IT size,Args... args) {
	extern __shared__ uchar shared[];
	const IT step=gridDim.x<<BDIML2;
	IT bidx=blockIdx.x<<BDIML2;
	IT nbidx=step+bidx;
	IT idx=bidx+threadIdx.x;
	if(nbidx<size) {
		for(;nbidx<size;nbidx+=step,idx+=step,bidx+=step)
			fn_T::fn(idx,size,shared,args...);
	}
	if(bidx<size)
		fn_T::fn(idx,size,shared,args...);

}
}
}
}
}
}
#endif
