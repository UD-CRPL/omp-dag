#ifndef __SPARSE_HANDLER_MATRIX_FORMATS_H__
#define __SPARSE_HANDLER_MATRIX_FORMATS_H__

#include "sparse-cxs.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T,typename IT=int> struct MatrixCXSHandler {
	IT* __ptr__=nullptr;
	IT* __indx__=nullptr;
	T*  __values__=nullptr;
	IT __rows__=0;
	IT __cols__=0;
	IT __nzv__=0;

	MatrixCXSHandler();
	template <typename Allocator,SparseCXSFormat frmt> MatrixCXSHandler(const MatrixCXS<T,IT,Allocator,frmt>& matrix);
	~MatrixCXSHandler();

	template <typename Allocator,SparseCXSFormat frmt> MatrixCXSHandler& operator=(const MatrixCXS<T,IT,Allocator,frmt>& matrix);
	template <typename Allocator,SparseCXSFormat frmt> MatrixCXSHandler& operator<<(const MatrixCXS<T,IT,Allocator,frmt>& matrix);

	__attr_fiffhd__ IT* ptr() const;
	__attr_fiffhd__ IT* indxs() const;
	__attr_fiffhd__ T* values() const;

	__attr_fiffhd__ IT n() const;
	__attr_fiffhd__ IT m() const;
	__attr_fiffhd__ IT rows() const;
	__attr_fiffhd__ IT cols() const;
	__attr_fiffhd__ IT nzv() const;

	template <typename ITT> __attr_fiffhd__ IT& ptr(ITT i);
	template <typename ITT> __attr_fiffhd__ const IT& ptr(ITT i) const;
	template <typename ITT> __attr_fiffhd__ IT& indxs(ITT i);
	template <typename ITT> __attr_fiffhd__ const IT& indxs(ITT i) const;
	template <typename ITT> __attr_fiffhd__ T& values(ITT i);
	template <typename ITT> __attr_fiffhd__ const T& values(ITT i) const;
	template <ReadMode readMode=read_only,typename ITT=IT> __attr_fiffhd__ T readPtr(ITT i) const;
	template <ReadMode readMode=read_only,typename ITT=IT> __attr_fiffhd__ T readIndxs(ITT i) const;
	template <ReadMode readMode=read_only,typename ITT=IT> __attr_fiffhd__ T readValues(ITT i) const;
};

template <typename T,typename IT>
inline MatrixCXSHandler<T,IT>::MatrixCXSHandler() {
}
template <typename T,typename IT> template <typename Allocator,SparseCXSFormat frmt>
inline MatrixCXSHandler<T,IT>::MatrixCXSHandler(const MatrixCXS<T,IT,Allocator,frmt>& matrix) {
	__ptr__=matrix.ptr();
	__indx__=matrix.indxs();
	__values__=matrix.values();
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__nzv__=matrix.nzv();
}
template <typename T,typename IT>
inline MatrixCXSHandler<T,IT>::~MatrixCXSHandler() {
	__ptr__=nullptr;
	__indx__=nullptr;
	__values__=nullptr;
	__rows__=0;
	__cols__=0;
	__nzv__=0;
}

template <typename T,typename IT> template <typename Allocator,SparseCXSFormat frmt>
inline MatrixCXSHandler<T,IT>& MatrixCXSHandler<T,IT>::operator=(const MatrixCXS<T,IT,Allocator,frmt>& matrix) {
	__ptr__=matrix.ptr();
	__indx__=matrix.indxs();
	__values__=matrix.values();
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__nzv__=matrix.nzv();
	return *this;
}
template <typename T,typename IT> template <typename Allocator,SparseCXSFormat frmt>
inline MatrixCXSHandler<T,IT>& MatrixCXSHandler<T,IT>::operator<<(const MatrixCXS<T,IT,Allocator,frmt>& matrix) {
	__ptr__=matrix.ptr();
	__indx__=matrix.indxs();
	__values__=matrix.values();
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__nzv__=matrix.nzv();
	return *this;
}

template <typename T,typename IT>
__attr_fiffhd__ IT* MatrixCXSHandler<T,IT>::ptr() const {
	return __ptr__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT* MatrixCXSHandler<T,IT>::indxs() const {
	return __indx__;
}
template <typename T,typename IT>
__attr_fiffhd__ T* MatrixCXSHandler<T,IT>::values() const {
	return __values__;
}

template <typename T,typename IT>
__attr_fiffhd__ IT MatrixCXSHandler<T,IT>::n() const {
	return __rows__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixCXSHandler<T,IT>::m() const {
	return __cols__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixCXSHandler<T,IT>::rows() const {
	return __rows__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixCXSHandler<T,IT>::cols() const {
	return __cols__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixCXSHandler<T,IT>::nzv() const {
	return __nzv__;
}

template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ IT& MatrixCXSHandler<T,IT>::ptr(ITT i) {
	return __ptr__[i];
}
template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ IT& MatrixCXSHandler<T,IT>::indxs(ITT i) {
	return __indx__[i];
}
template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ T& MatrixCXSHandler<T,IT>::values(ITT i) {
	return __values__[i];
}
template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ const IT& MatrixCXSHandler<T,IT>::ptr(ITT i) const {
	return __ptr__[i];
}
template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ const IT& MatrixCXSHandler<T,IT>::indxs(ITT i) const {
	return __indx__[i];
}
template <typename T,typename IT> template <typename ITT>
__attr_fiffhd__ const T& MatrixCXSHandler<T,IT>::values(ITT i) const {
	return __values__[i];
}
template <typename T,typename IT> template <ReadMode readMode,typename ITT>
__attr_fiffhd__ T MatrixCXSHandler<T,IT>::readPtr(ITT i) const {
	return read_memory<readMode>(__ptr__+i);
}
template <typename T,typename IT> template <ReadMode readMode,typename ITT>
__attr_fiffhd__ T MatrixCXSHandler<T,IT>::readIndxs(ITT i) const {
	return read_memory<readMode>(__indx__+i);
}
template <typename T,typename IT> template <ReadMode readMode,typename ITT>
__attr_fiffhd__ T MatrixCXSHandler<T,IT>::readValues(ITT i) const {
	return read_memory<readMode>(__values__+i);
}
}
}
}
#endif
