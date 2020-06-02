#ifndef __ARRAY_HANDLER_DATA_STRUCTURES_CORE_H__
#define __ARRAY_HANDLER_DATA_STRUCTURES_CORE_H__

#include "array.h"
#include "../types/type-rw.h"

namespace __core__ {
namespace __data_structures__ {
template <typename T,typename IT=int> struct ArrayHandler {
	T* __data__=nullptr;
	IT __size__=0;

	ArrayHandler();
	template <typename Allocator> ArrayHandler(const Array<T,Allocator>& array);
	~ArrayHandler();

	T* __host_device__ operator*() const;

	template <typename Allocator> ArrayHandler& operator=(const Array<T,Allocator>& array);
	template <typename Allocator> ArrayHandler& operator<<(const Array<T,Allocator>& array);

	template <typename ITT=IT> __attr_fiffhd__ T& operator[](ITT i);
	template <typename ITT=IT> __attr_fiffhd__ const T& operator[](ITT i) const;

	template <typename TT=T> __attr_fiffhd__ TT* data() const;
	template <typename TT=T,typename ITT=IT> __attr_fiffhd__ TT& data(ITT i);
	template <typename TT=T,typename ITT=IT> __attr_fiffhd__ const TT& data(ITT i) const;

	__attr_fiffhd__ IT n() const;
	__attr_fiffhd__ IT size() const;

	template <__core__::__type__::ReadMode readMode=__core__::__type__::read_only,typename TT=T,typename ITT=IT> __attr_fiffhd__ TT read(IT i) const;
};
template <typename T,typename IT>
inline ArrayHandler<T,IT>::ArrayHandler() {
}
template <typename T,typename IT> template <typename Allocator>
inline ArrayHandler<T,IT>::ArrayHandler(const Array<T,Allocator>& array) {
	__data__=*array;
	__size__=array.size();
}
template <typename T,typename IT>
inline ArrayHandler<T,IT>::~ArrayHandler() {
	__data__=nullptr;
	__size__=0;
}
template <typename T,typename IT>
inline T* ArrayHandler<T,IT>::operator *() const {
	return __data__;
}
template <typename T,typename IT> template <typename Allocator>
inline ArrayHandler<T,IT>& ArrayHandler<T,IT>::operator =(const Array<T,Allocator>& array) {
	__data__=*array;
	__size__=array.size();
}
template <typename T,typename IT> template <typename Allocator>
inline ArrayHandler<T,IT>& ArrayHandler<T,IT>::operator <<(const Array<T,Allocator>& array) {
	__data__=*array;
	__size__=array.size();
}
template <typename T,typename IT> template <typename ITT>
inline T& ArrayHandler<T,IT>::operator [](ITT i) {
	return __data__[i];
}
template <typename T,typename IT> template <typename ITT>
inline const T& ArrayHandler<T,IT>::operator [](ITT i) const {
	return __data__[i];
}
template <typename T,typename IT> template <typename TT>
inline TT* ArrayHandler<T,IT>::data() const {
	return reinterpret_cast<TT*>(__data__);
}
template <typename T,typename IT> template <typename TT,typename ITT>
inline TT& ArrayHandler<T,IT>::data(ITT i) {
	return *reinterpret_cast<TT*>(__data__+i);
}
template <typename T,typename IT> template <typename TT,typename ITT>
inline const TT& ArrayHandler<T,IT>::data(ITT i) const {
	return *reinterpret_cast<const TT*>(__data__+i);
}
template <typename T,typename IT>
inline IT ArrayHandler<T,IT>::n() const {
	return __size__;
}
template <typename T,typename IT>
inline IT ArrayHandler<T,IT>::size() const {
	return __size__;
}
template <typename T,typename IT> template <__core__:: __type__ ::ReadMode readMode,typename TT,typename ITT>
inline TT ArrayHandler<T,IT>::read(IT i) const {
	return __core__:: __type__ ::read_memory<readMode>(reinterpret_cast<const TT*>(__data__+i));
}
}
}

#endif
