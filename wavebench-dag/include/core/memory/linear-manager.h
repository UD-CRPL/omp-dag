#ifndef __LINEAR_MANAGER_MEMORY_CORE_H__
#define __LINEAR_MANAGER_MEMORY_CORE_H__

#include "../macros/compiler.h"
#include "../enum-definitions.h"
#include "../debug/debug.h"
#include "../meta/meta.h"

#include "memory-types.h"
#include "util.h"
#include "memory-alloc.h"
#include "memory-free.h"
#include "memory-set.h"
#include "memory-copy.h"

namespace __core__ {
namespace __memory__ {
template <typename MT> class LinearMemoryManager {
public:
	typedef MT memory_type;
	static constexpr DeviceType location=memory_type::location;
	LinearMemoryManager(int dev=-1);
	LinearMemoryManager(const LinearMemoryManager&);
	LinearMemoryManager(LinearMemoryManager&& manager);
	~LinearMemoryManager();

	LinearMemoryManager& operator=(const LinearMemoryManager&);
	LinearMemoryManager& operator=(LinearMemoryManager&& manager);

	int device() const;
	bool consistent_device(int dev) const;

	void free();

	template <typename MTT,typename T=void,enable_IT<is_same_CE<MT,MTT>()> = 0> void attach(T* array,std::size_t allocatedSize,int sdev);
	void attach(LinearMemoryManager&& manager);
	void detach();

	template <typename T=void,typename MTT=MT,enable_IT<is_same_CE<MTT,void>()> = 0> static
	T* allocate(std::size_t size,int dev=-1);
	template <typename T=void,typename MTT=MT,enable_IT<!is_same_CE<MTT,void>()> = 0> static
	T* allocate(std::size_t size,int dev=-1);
	template <typename T=void,typename MTT=MT,enable_IT<is_same_CE<MTT,void>()> = 0> static
	T* callocate(std::size_t size,int dev=-1,StreamType stream=(StreamType)0);
	template <typename T=void,typename MTT=MT,enable_IT<!is_same_CE<MTT,void>()> = 0> static
	T* callocate(std::size_t size,int dev=-1,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,typename T=void,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()> = 0> static
	T* reallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,typename T=void,enable_IT<!(is_same_CE<MDT,void>()||is_same_CE<MST,void>())> = 0> static
	T* reallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,typename T=void,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()> = 0> static
	T* recallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,typename T=void,enable_IT<!(is_same_CE<MDT,void>()||is_same_CE<MST,void>())> = 0> static
	T* recallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MTT=MT,typename T=void,enable_IT<is_same_CE<MTT,void>()> = 0> static
	void deallocate(T* ptr,int dev=-1);
	template <typename MTT=MT,typename T=void,enable_IT<!is_same_CE<MTT,void>()> = 0> static
	void deallocate(T* ptr,int dev=-1);

	template <typename MTT=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<is_same_CE<MTT,void>()> = 0> static
	void set(T* ptr,std::size_t size,int value,StreamType stream=(StreamType)0);
	template <typename MTT=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<!is_same_CE<MTT,void>()> = 0> static
	void set(T* ptr,std::size_t size,int value,StreamType stream=(StreamType)0);
	template <typename MTT=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<is_same_CE<MTT,void>()> = 0> static
	void set(T* ptr,std::size_t size,int value,int dev,StreamType stream=(StreamType)0);
	template <typename MTT=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<!is_same_CE<MTT,void>()> = 0> static
	void set(T* ptr,std::size_t size,int value,int dev,StreamType stream=(StreamType)0);

	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,int dev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,int dev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())&&(__memory_private__::transfer_type<MDT,MST>()==GPU_GPU)> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream=(StreamType)0);
	template <typename MDT=MT,typename MST=MT,SyncBehaviorType sync_behavior=ASYNC,typename T=void,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())&&(__memory_private__::transfer_type<MDT,MST>()!=GPU_GPU)> = 0> static
	void copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream=(StreamType)0);
};
template <typename MT> LinearMemoryManager<MT>::LinearMemoryManager(int dev) {
}
template <typename MT> LinearMemoryManager<MT>::LinearMemoryManager(const LinearMemoryManager&) {
}
template <typename MT> LinearMemoryManager<MT>::LinearMemoryManager(LinearMemoryManager&& manager) {
}
template <typename MT> LinearMemoryManager<MT>::~LinearMemoryManager() {
}

template <typename MT> LinearMemoryManager<MT>& LinearMemoryManager<MT>::operator=(const LinearMemoryManager&) {
	return *this;
}
template <typename MT> LinearMemoryManager<MT>& LinearMemoryManager<MT>::operator=(LinearMemoryManager&& manager) {
	return *this;
}

template <typename MT> int LinearMemoryManager<MT>::device() const {
	return -1;
}
template <typename MT> bool LinearMemoryManager<MT>::consistent_device(int dev) const {
	if(device()<0)
		return true;
	else
		return dev==device();
}

template <typename MT> void LinearMemoryManager<MT>::free() {
}
template <typename MT> template <typename MTT,typename T,enable_IT<is_same_CE<MT,MTT>()>> void LinearMemoryManager<MT>::attach(T* array,std::size_t allocatedSize,int sdev) {
}
template <typename MT> void LinearMemoryManager<MT>::attach(LinearMemoryManager&& manager) {
}
template <typename MT> void LinearMemoryManager<MT>::detach() {
}

template <typename MT> template <typename T,typename MTT,enable_IT<is_same_CE<MTT,void>()>>
T* LinearMemoryManager<MT>::allocate(std::size_t size,int dev) {
}
template <typename MT> template <typename T,typename MTT,enable_IT<!is_same_CE<MTT,void>()>>
T* LinearMemoryManager<MT>::allocate(std::size_t size,int dev) {
	T* ptr=nullptr;
	if(size>0)
		__malloc__<MTT>(ptr,size,dev);
	return ptr;
}
template <typename MT> template <typename T,typename MTT,enable_IT<is_same_CE<MTT,void>()>>
T* LinearMemoryManager<MT>::callocate(std::size_t size,int dev,StreamType stream) {
}
template <typename MT> template <typename T,typename MTT,enable_IT<!is_same_CE<MTT,void>()>>
T* LinearMemoryManager<MT>::callocate(std::size_t size,int dev,StreamType stream) {
	T* ptr=allocate<T,MTT>(size,dev);
	set<MTT,SYNC>(ptr,size,0,dev,stream);
	return ptr;
}
template <typename MT> template <typename MDT,typename MST,typename T,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()>>
T* LinearMemoryManager<MT>::reallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream) {
}
template <typename MT> template <typename MDT,typename MST,typename T,enable_IT<!(is_same_CE<MDT,void>()||is_same_CE<MST,void>())>>
T* LinearMemoryManager<MT>::reallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream) {
	T* tmp=allocate<T,MDT>(size,ddev);
	copy<MDT,MST>(tmp,sptr,std::min(size,ssize),ddev,sdev,stream);
	deallocate<MST>(sptr,ssize);
	return tmp;
}
template <typename MT> template <typename MDT,typename MST,typename T,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()>>
T* LinearMemoryManager<MT>::recallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream) {
}
template <typename MT> template <typename MDT,typename MST,typename T,enable_IT<!(is_same_CE<MDT,void>()||is_same_CE<MST,void>())>>
T* LinearMemoryManager<MT>::recallocate(std::size_t size,T* sptr,std::size_t ssize,int ddev,int sdev,StreamType stream) {
	T* tmp=allocate<T,MDT>(size,ddev);
	set<MDT,SYNC>(tmp,size,0,ddev,stream);
	copy<MDT,MST>(tmp,sptr,std::min(size,ssize),ddev,sdev,stream);
	deallocate<MST>(sptr,ssize);
	return tmp;
}
template <typename MT> template <typename MTT,typename T,enable_IT<is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::deallocate(T* ptr,int dev) {
}
template <typename MT> template <typename MTT,typename T,enable_IT<!is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::deallocate(T* ptr,int dev) {
	if(ptr!=nullptr) {
		if(dev>=0)
			__free__<MTT>(ptr,dev);
		else
			__free__<MTT>(ptr);
	}
}

template <typename MT> template <typename MTT,SyncBehaviorType sync_behavior,typename T,enable_IT<is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::set(T* ptr,std::size_t size,int value,StreamType stream) {
}
template <typename MT> template <typename MTT,SyncBehaviorType sync_behavior,typename T,enable_IT<!is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::set(T* ptr,std::size_t size,int value,StreamType stream) {
	set<MTT,sync_behavior>(ptr,size,value,-1,stream);
}
template <typename MT> template <typename MTT,SyncBehaviorType sync_behavior,typename T,enable_IT<is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::set(T* ptr,std::size_t size,int value,int dev,StreamType stream) {
}
template <typename MT> template <typename MTT,SyncBehaviorType sync_behavior,typename T,enable_IT<!is_same_CE<MTT,void>()>>
void LinearMemoryManager<MT>::set(T* ptr,std::size_t size,int value,int dev,StreamType stream) {
	if(ptr!=nullptr&&size>0)
		__memset__<MTT,sync_behavior>(ptr,value,size,dev,stream);
}

template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,StreamType stream) {
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,StreamType stream) {
	copy<MDT,MST,sync_behavior>(dptr,sptr,size,-1,stream);
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,int dev,StreamType stream) {
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,int dev,StreamType stream) {
	if(sptr!=nullptr&&dptr!=nullptr&&size>0)
		__memcpy__<MDT,MST,sync_behavior>(dptr,sptr,size,dev,stream);
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<is_same_CE<MDT,void>()||is_same_CE<MST,void>()>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream) {
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())&&(__memory_private__::transfer_type<MDT,MST>()==GPU_GPU)>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream) {
	if(sptr!=nullptr&&dptr!=nullptr&&size>0) {
		if(sdev!=ddev)
			__memcpy__<MDT,MST,sync_behavior>(dptr,sptr,size,ddev,sdev,stream);
		else
			__memcpy__<MDT,MST,sync_behavior>(dptr,sptr,size,ddev,stream);
	}
}
template <typename MT> template <typename MDT,typename MST,SyncBehaviorType sync_behavior,typename T,enable_IT<(!is_same_CE<MDT,void>())&&(!is_same_CE<MST,void>())&&(__memory_private__::transfer_type<MDT,MST>()!=GPU_GPU)>>
void LinearMemoryManager<MT>::copy(T* dptr,T* sptr,std::size_t size,int ddev,int sdev,StreamType stream) {
	if(sptr!=nullptr&&dptr!=nullptr&&size>0) {
		__memcpy__<MDT,MST,sync_behavior>(dptr,sptr,size,ddev,stream);
	}
}

template <MemoryType type=NORMAL_MEM,bool portable=false,bool mapped=false,bool write_combined=false> using cpuAllocator_T=LinearMemoryManager<cpu_memory_T<type,portable,mapped,write_combined>>;
template <bool portable=false,bool mapped=false,bool write_combined=false> using pinnedAllocator_T=LinearMemoryManager<pinned_memory_T<portable,mapped,write_combined>>;
template <bool attach_global=true> using managedAllocator_T=LinearMemoryManager<managed_memory_T<attach_global>>;
using cpuAllocator=cpuAllocator_T<>;
using pinnedAllocator=pinnedAllocator_T<>;
using managedAllocator=managedAllocator_T<>;
using gpuAllocator=LinearMemoryManager<gpu_memory>;
}
}
#endif
