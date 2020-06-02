#ifndef __MEMORY_TYPES_MEMORY_CORE_H__
#define __MEMORY_TYPES_MEMORY_CORE_H__

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
#endif

#include "../enum-definitions.h"

namespace __core__ {
namespace __memory__ {
template <DeviceType memory_location,MemoryType TYPE,bool PORTABLE=false,bool MAPPED=false,bool WRITE_COMBINED=false,bool ATTACH_GLOBAL=false> struct __MemoryType__ {
	static_assert((!((memory_location==HOST)&&(TYPE==MANAGED_MEM))),"Invalid combination of parameters!!!");
	static_assert((!((memory_location==DEVICE)&&(TYPE==MANAGED_MEM))),"Invalid combination of parameters!!!");
	static_assert((!((memory_location==DEVICE)&&(TYPE==PINNED_MEM))),"Invalid combination of parameters!!!");
	static_assert((!((memory_location==MANAGED)&&(TYPE==NORMAL_MEM))),"Invalid combination of parameters!!!");
	static_assert((!((memory_location==MANAGED)&&(TYPE==PINNED_MEM))),"Invalid combination of parameters!!!");

	static constexpr DeviceType location=memory_location;
	static constexpr MemoryType type=(location==DEVICE)?NORMAL_MEM:((location==MANAGED)?MANAGED_MEM:TYPE);
	static constexpr bool is_pinned=(type==PINNED_MEM);
	static constexpr bool is_managed=(type==MANAGED_MEM);
	static constexpr bool is_portable=(type==PINNED_MEM)?PORTABLE:false;
	static constexpr bool is_mapped=(type==MANAGED_MEM)?true:((type==PINNED_MEM)?MAPPED:false);
	static constexpr bool is_write_combined=(type==PINNED_MEM)?WRITE_COMBINED:false;
	static constexpr bool is_attach_global=(type==MANAGED_MEM)?ATTACH_GLOBAL:false;
	static constexpr bool is_attach_host=(type==MANAGED_MEM)?(!ATTACH_GLOBAL):false;
#if defined(__CUDARUNTIMEQ__)
	static constexpr int managed_F=(type==MANAGED_MEM)?((ATTACH_GLOBAL==true)?cudaMemAttachGlobal:cudaMemAttachHost):0;
	static constexpr int host_alloc_F=(type==NORMAL_MEM)?0:((is_portable?cudaHostAllocPortable:0)|(is_mapped?cudaHostAllocMapped:0)|(is_write_combined?cudaHostAllocWriteCombined:0));
#else
	static constexpr int managed_F=0;
	static constexpr int host_alloc_F=0;
#endif
};

template <MemoryType type=NORMAL_MEM,bool portable=false,bool mapped=false,bool write_combined=false> using cpu_memory_T=__MemoryType__<HOST,type,portable,mapped,write_combined,false>;
template <bool portable=false,bool mapped=false,bool write_combined=false> using pinned_memory_T=__MemoryType__<HOST,PINNED_MEM,portable,mapped,write_combined,false>;
template <bool attach_global=true> using managed_memory_T=__MemoryType__<MANAGED,MANAGED_MEM,false,true,false,attach_global>;

using cpu_memory=cpu_memory_T<>;
using pinned_memory=pinned_memory_T<>;
using managed_memory=managed_memory_T<>;
using gpu_memory=__MemoryType__<DEVICE,NORMAL_MEM,false,false,false,false>;
}
}
#endif
