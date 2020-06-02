#ifndef __ARRAY_DATA_STRUCTURES_CORE_H__
#define __ARRAY_DATA_STRUCTURES_CORE_H__

#include "../memory/memory.h"
#include "../util/util.h"

namespace __core__ {
namespace __data_structures__ {
template <typename T,typename Allocator> class Array {
public:
	typedef T type;
	typedef Allocator allocator_type;
	typedef typename Allocator::memory_type memory_type;
	static const std::size_t npos=__numeric_limits__<size_t>::max;
private:
	T* __data__=nullptr;
	std::size_t __size__=0;
	std::size_t __capacity__=0;
	std::size_t __allocated_size__=0;
	int ___device___=-1;
	Allocator __allocator__;
	template <typename TT,typename allocatorType_T> friend class Array;
	void __reserve__(std::size_t size);
	void __set_device__(int dev);
public:
	Array(int dev=-1);
	Array(Array&& array);
	Array(const Array& array);
	Array(const Allocator& allocator);
	Array(const Array& array,const Allocator& allocator);
	Array(std::size_t size,const Allocator& allocator=Allocator());
	Array(std::size_t size,int dev,const Allocator& allocator=Allocator());
	Array(std::size_t size,int dev,int val,StreamType stream=(StreamType)0,const Allocator& allocator=Allocator());
	template <typename allocatorType_T> Array(const Array<T,allocatorType_T>& array,int dev=-1,StreamType stream=(StreamType)0,const Allocator& allocator=Allocator());
	~Array();

	Array& operator=(Array&& array);
	Array& operator=(const Array& array);
	template <typename allocatorType_T> Array& operator=(const Array<T,allocatorType_T>& array);

	T* operator*() const;
	T& operator[](std::size_t i);
	const T& operator[](std::size_t i) const;

	void set_allocator(Allocator&& allocator);
	void set_allocator(const Allocator& allocator);
	Allocator& get_allocator();
	const Allocator& get_allocator() const;

	template <typename U=T> U* data() const;

	template <typename U=T> U& at(std::size_t i);
	template <typename U=T> const U& at(std::size_t i) const;

	int device() const;
	bool empty() const;
	bool consistent() const;
	std::size_t size() const;
	std::size_t capacity() const;

	void free();
	void free_allocator();
	void clear();
	void expand();
	void reserve(std::size_t size,StreamType stream=(StreamType)0);
	void reserve(std::size_t size,int val,StreamType stream);
	void resize(std::size_t size,int val=0,StreamType stream=(StreamType)0);

	void set(int val=0,StreamType stream=(StreamType)0);
	void set(int val,std::size_t begin,std::size_t len,StreamType stream=(StreamType)0);

	void detach();
	template <typename memory_T> void attach(T* array,std::size_t size,std::size_t capacity,int sdev=-1);

	Array& move(Array&& array);
	template <typename memory_T,bool check=true,enable_IT<check> = 0> Array& import(T* array,std::size_t begin,std::size_t size,int sdev=-1,StreamType stream=(StreamType)0);
	template <typename memory_T,bool check=true,enable_IT<!check> = 0> Array& import(T* array,std::size_t begin,std::size_t size,int sdev=-1,StreamType stream=(StreamType)0);
	template <typename memory_T,bool check=true,enable_IT<check> = 0> Array& import(T* array,std::size_t size,int sdev=-1,StreamType stream=(StreamType)0);
	template <typename memory_T,bool check=true,enable_IT<!check> = 0> Array& import(T* array,std::size_t size,int sdev=-1,StreamType stream=(StreamType)0);
	template <bool check=true,typename allocatorType_T=void,enable_IT<check> = 0>
	Array& import(const Array<T,allocatorType_T>& array,std::size_t sbegin,std::size_t dbegin=0,std::size_t size=npos,StreamType stream=(StreamType)0);
	template <bool check=true,typename allocatorType_T=void,enable_IT<!check> = 0>
	Array& import(const Array<T,allocatorType_T>& array,std::size_t sbegin,std::size_t dbegin=0,std::size_t size=npos,StreamType stream=(StreamType)0);
	template <bool check=true,typename allocatorType_T=void,enable_IT<check> = 0> Array& import(const Array<T,allocatorType_T>& array,StreamType stream=(StreamType)0);
	template <bool check=true,typename allocatorType_T=void,enable_IT<!check> = 0> Array& import(const Array<T,allocatorType_T>& array,StreamType stream=(StreamType)0);
};
template <typename T,typename Allocator>
Array<T,Allocator>::Array(int dev): __data__(nullptr), __allocator__(Allocator(dev)) {
	__set_device__(dev);
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(Array&& array) {
	move(std::forward<Array>(array));
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(const Array& array): __allocator__(Allocator()) {
	__set_device__(-1);
	__reserve__(array.size());
	import<false>(array);
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(const Allocator& allocator): __allocator__(allocator) {
	__set_device__(-1);
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(const Array& array,const Allocator& allocator): Array(allocator) {
	__reserve__(array.size());
	import<false>(array);
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(std::size_t size,const Allocator& allocator): Array(allocator) {
	__reserve__(size);
	set();
}
template <typename T,typename Allocator>
Array<T,Allocator>::Array(std::size_t size,int dev,const Allocator& allocator): __allocator__(allocator) {
	__set_device__(dev);
	__reserve__(size);
	set();
}
template <typename T,typename Allocator> 
Array<T,Allocator>::Array(std::size_t size,int dev,int val,StreamType stream,const Allocator& allocator): __allocator__(allocator) {
	__set_device__(dev);
	__reserve__(size);
	set(val,stream);
}
template <typename T,typename Allocator> template <typename allocatorType_T>
Array<T,Allocator>::Array(const Array<T,allocatorType_T>& array,int dev,StreamType stream,const Allocator& allocator): __allocator__(allocator) {
	__set_device__(dev);
	__reserve__(array.size());
	import<false>(array);
}
template <typename T,typename Allocator>
Array<T,Allocator>::~Array() {
	free();
	free_allocator();
}

template <typename T,typename Allocator>
Array<T,Allocator>& Array<T,Allocator>::operator =(Array&& array) {
	return move(std::forward<Array>(array));
}
template <typename T,typename Allocator>
Array<T,Allocator>& Array<T,Allocator>::operator =(const Array& array) {
	return import<true>(array);
}
template <typename T,typename Allocator> template <typename allocatorType_T>
Array<T,Allocator>& Array<T,Allocator>::operator =(const Array<T,allocatorType_T>& array) {
	return import<true>(array);
}

template <typename T,typename Allocator>
T* Array<T,Allocator>::operator *() const {
	return __data__;
}
template <typename T,typename Allocator>
T& Array<T,Allocator>::operator [](std::size_t i) {
	return __data__[i];
}
template <typename T,typename Allocator>
const T& Array<T,Allocator>::operator [](std::size_t i) const {
	return __data__[i];
}

template <typename T,typename Allocator>
void Array<T,Allocator>::__reserve__(std::size_t size) {
	__size__=size;
	__capacity__=__size__;
	__allocated_size__=__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>();
	__data__=__allocator__.template allocate<T,memory_type>(__capacity__,___device___);
}
template <typename T,typename Allocator>
void Array<T,Allocator>::__set_device__(int dev) {
	if(__allocator__.device()>=0)
		___device___=__allocator__.device();
	else if(dev>=0)
		___device___=__core__::__util__::__cuda__::valid_device(dev);
	else
		___device___=get_device();
}

template <typename T,typename Allocator>
void Array<T,Allocator>::set_allocator(Allocator&& allocator) {
	__allocator__=std::forward<Allocator>(allocator);
}
template <typename T,typename Allocator>
void Array<T,Allocator>::set_allocator(const Allocator& allocator) {
	__allocator__=allocator;
}
template <typename T,typename Allocator>
Allocator& Array<T,Allocator>::get_allocator() {
	return __allocator__;
}
template <typename T,typename Allocator>
const Allocator& Array<T,Allocator>::get_allocator() const {
	return __allocator__;
}

template <typename T,typename Allocator> template <typename U>
U* Array<T,Allocator>::data() const {
	return reinterpret_cast<U*>(__data__);
}
template <typename T,typename Allocator> template <typename U>
U& Array<T,Allocator>::at(std::size_t i) {
	if(i*__memory__::__memory_private__::__sizeof__<U,std::size_t>()<__size__*__memory__::__memory_private__::__sizeof__<T,std::size_t>())
		return reinterpret_cast<U*>(__data__)[i];
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}
template <typename T,typename Allocator> template <typename U>
const U& Array<T,Allocator>::at(std::size_t i) const {
	if(i*__memory__::__memory_private__::__sizeof__<U,std::size_t>()<__size__*__memory__::__memory_private__::__sizeof__<T,std::size_t>())
		return reinterpret_cast<U*>(__data__)[i];
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}

template <typename T,typename Allocator>
int Array<T,Allocator>::device() const {
	return ___device___;
}
template <typename T,typename Allocator>
bool Array<T,Allocator>::empty() const {
	return (__size__==0);
}
template <typename T,typename Allocator>
bool Array<T,Allocator>::consistent() const {
	return (__size__<=__capacity__)&&(__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>()<=__allocated_size__)&&
			(__data__==nullptr?(__allocated_size__==0):(__allocated_size__!=0));
}
template <typename T,typename Allocator>
std::size_t Array<T,Allocator>::size() const {
	return __size__;
}
template <typename T,typename Allocator>
std::size_t Array<T,Allocator>::capacity() const {
	return __capacity__;
}

template <typename T,typename Allocator>
void Array<T,Allocator>::free() {
	__size__=0;
	__capacity__=0;
	__allocated_size__=0;
	___device___=-1;
	__allocator__.template deallocate<memory_type,T>(__data__);
	__data__=nullptr;
}
template <typename T,typename Allocator>
void Array<T,Allocator>::free_allocator() {
	__allocator__.free();
}
template <typename T,typename Allocator>
void Array<T,Allocator>::clear() {
	__size__=0;
}
template <typename T,typename Allocator>
void Array<T,Allocator>::expand() {
	__size__=__capacity__;
}
template <typename T,typename Allocator> 
void Array<T,Allocator>::reserve(std::size_t size,StreamType stream) {
	if(size<=__capacity__)
		return ;
	else {
		if(___device___<0)
			__set_device__(-1);
		__data__=__allocator__.template reallocate<memory_type,memory_type>(size,__data__,__size__,___device___,___device___,stream);
		__capacity__=size;
		__allocated_size__=__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>();
	}
}
template <typename T,typename Allocator> 
void Array<T,Allocator>::reserve(std::size_t size,int val,StreamType stream) {
	if(size<=__capacity__)
		return ;
	else {
		if(___device___<0)
			__set_device__(-1);
		__data__=__allocator__.template reallocate<memory_type,memory_type>(size,__data__,__size__,___device___,___device___,stream);
		set(val,__size__,size-__size__,stream);
		__capacity__=size;
		__allocated_size__=__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>();
	}
}
template <typename T,typename Allocator> 
void Array<T,Allocator>::resize(std::size_t size,int val,StreamType stream) {
	if(size<=__capacity__)
		__size__=size;
	else {
		if(___device___<0)
			__set_device__(-1);
		__data__=__allocator__.template reallocate<memory_type,memory_type>(size,__data__,__size__,___device___,___device___,stream);
		set(val,__size__,size-__size__,stream);
		__size__=size;
		__capacity__=__size__;
		__allocated_size__=__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>();
	}
}

template <typename T,typename Allocator> 
void Array<T,Allocator>::set(int val,StreamType stream) {
	__allocator__.template set<memory_type,ASYNC>(__data__,__capacity__,val,stream);
}
template <typename T,typename Allocator> 
void Array<T,Allocator>::set(int val,std::size_t begin,std::size_t len,StreamType stream) {
	if(((begin+len)<=__capacity__)&&((begin+len)>begin))
		__allocator__.template set<memory_type,ASYNC>(__data__+begin,len,val,stream);
}

template <typename T,typename Allocator>
void Array<T,Allocator>::detach() {
	__size__=0;
	__capacity__=0;
	__allocated_size__=0;
	___device___=-1;
	__data__=nullptr;
}
template <typename T,typename Allocator> template <typename memory_T>
void Array<T,Allocator>::attach(T* array,std::size_t size,std::size_t capacity,int sdev) {
	if(array!=nullptr&&size>0&&size<=capacity) {
		free();
		__data__=array;
		__size__=size;
		__capacity__=capacity;
		__allocated_size__=__capacity__*__memory__::__memory_private__::__sizeof__<T,std::size_t>();
		___device___=sdev;
		__allocator__.template attach<memory_T>(__data__,__allocated_size__,___device___);
	}
}

template <typename T,typename Allocator>
Array<T,Allocator>& Array<T,Allocator>::move(Array&& array) {
	__memory__::move(__data__,array.__data__);
	__memory__::move(__size__,array.__size__);
	__memory__::move(__capacity__,array.__capacity__);
	__memory__::move(__allocated_size__,array.__allocated_size__);
	array.__data__=nullptr;
	___device___=array.___device___;
	array.___device___=-1;
	__allocator__=std::move(array.__allocator__);
	if(!__allocator__.consistent_device(___device___)) {
		error(true,"The devices between the array and the allocator are inconsistent.",RUNTIME_ERROR,throw_error);
	}
	return *this;
}

template <typename T,typename Allocator> template <typename memory_T,bool check,enable_IT<check>>
Array<T,Allocator>& Array<T,Allocator>::import(T* array,std::size_t begin,std::size_t size,int sdev,StreamType stream) {
	if(size==npos) {
		error(true,"Invalid memory region for the source array.",RUNTIME_ERROR,throw_error);
	}
	std::size_t nsize=begin+size;
	if(nsize<=__capacity__) {
		__size__=std::max(nsize,__size__);
	}
	else {
		if(begin==0) {
			int dev=___device___;
			free();
			___device___=dev;
			reserve(nsize,stream);
			expand();
			return import<memory_T,false>(array,begin,size,sdev,stream);
		}
		else
			resize(nsize,0,stream);
	}
	return import<memory_T,false>(array,begin,size,sdev,stream);
}
template <typename T,typename Allocator> template <typename memory_T,bool check,enable_IT<!check>>
Array<T,Allocator>& Array<T,Allocator>::import(T* array,std::size_t begin,std::size_t size,int sdev,StreamType stream) {
	__allocator__.template copy<memory_type,memory_T,ASYNC>(__data__+begin,array,size,___device___,sdev,stream);
	return *this;
}
template <typename T,typename Allocator> template <typename memory_T,bool check,enable_IT<check>>
Array<T,Allocator>& Array<T,Allocator>::import(T* array,std::size_t size,int sdev,StreamType stream) {
	return import<memory_T,true>(array,0,size,sdev,stream);
}
template <typename T,typename Allocator> template <typename memory_T,bool check,enable_IT<!check>>
Array<T,Allocator>& Array<T,Allocator>::import(T* array,std::size_t size,int sdev,StreamType stream) {
	return import<memory_T,false>(array,0,size,sdev,stream);
}

template <typename T,typename Allocator> template <bool check,typename allocatorType_T,enable_IT<check>>
Array<T,Allocator>& Array<T,Allocator>::import(const Array<T,allocatorType_T>& array,std::size_t sbegin,std::size_t dbegin,std::size_t size,StreamType stream) {
	if(size==npos)
		size=array.size()>sbegin?(array.size()-sbegin):0;
	if(((sbegin+size)<=array.size())&&((sbegin+size)>sbegin)&&(size>0)) {
		import<typename allocatorType_T::memory_type,check>(array.data()+sbegin,dbegin,size,array.device(),stream);
	}
	else {
		error(true,"Invalid memory region for the source array.",RUNTIME_ERROR,throw_error);
	}
	return *this;
}
template <typename T,typename Allocator> template <bool check,typename allocatorType_T,enable_IT<!check>>
Array<T,Allocator>& Array<T,Allocator>::import(const Array<T,allocatorType_T>& array,std::size_t sbegin,std::size_t dbegin,std::size_t size,StreamType stream) {
	return import<typename allocatorType_T::memory_type,check>(array.data()+sbegin,dbegin,std::min(size,std::min(__size__-dbegin,array.size()-sbegin)),array.device(),stream);
}
template <typename T,typename Allocator> template <bool check,typename allocatorType_T,enable_IT<check>>
Array<T,Allocator>& Array<T,Allocator>::import(const Array<T,allocatorType_T>& array,StreamType stream) {
	return import<check>(array,0,0,array.size(),stream);
}
template <typename T,typename Allocator> template <bool check,typename allocatorType_T,enable_IT<!check>>
Array<T,Allocator>& Array<T,Allocator>::import(const Array<T,allocatorType_T>& array,StreamType stream) {
	return import<check>(array,0,0,array.size(),stream);
}
}
}
#endif
