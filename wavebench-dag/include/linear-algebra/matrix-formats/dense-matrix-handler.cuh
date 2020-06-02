#ifndef __DENSE_MATRIX_HANDLER_MATRIX_FORMATS_H__
#define __DENSE_MATRIX_HANDLER_MATRIX_FORMATS_H__

#include "dense-matrix.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T,typename IT=int> struct MatrixHandler {
	uchar* __data__=nullptr;
	IT __rows__=0;
	IT __cols__=0;
	IT __pitch__=0;

	MatrixHandler();
	template <typename Allocator,uint alignment> MatrixHandler(const Matrix<T,Allocator,alignment>& matrix);
	~MatrixHandler();
	uchar* __host_device__ operator*() const;

	template <typename Allocator,uint alignment> MatrixHandler& operator=(const Matrix<T,Allocator,alignment>& matrix);
	template <typename Allocator,uint alignment> MatrixHandler& operator<<(const Matrix<T,Allocator,alignment>& matrix);

	__attr_fiffhd__  T& operator()(IT i,IT j);
	__attr_fiffhd__ const T& operator()(IT i,IT j) const;
	__attr_fiffhd__ T& operator[](IT i);
	__attr_fiffhd__ const T& operator[](IT i) const;

	__attr_fiffhd__ IT n() const;
	__attr_fiffhd__ IT m() const;
	__attr_fiffhd__ IT rows() const;
	__attr_fiffhd__ IT cols() const;
	__attr_fiffhd__ IT pitch() const;

	template <ReadMode readMode=read_only> __attr_fiffhd__ T read(IT i,IT j) const;
	template <ReadMode readMode=read_only> __attr_fiffhd__ T read(IT i) const;
};

template <typename T,typename IT>
inline MatrixHandler<T,IT>::MatrixHandler() {
}
template <typename T,typename IT> template <typename Allocator,uint alignment>
inline MatrixHandler<T,IT>::MatrixHandler(const Matrix<T,Allocator,alignment>& matrix) {
	__data__=*matrix;
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__pitch__=matrix.pitch();
}
template <typename T,typename IT>
inline MatrixHandler<T,IT>::~MatrixHandler() {
	__data__=nullptr;
	__rows__=0;
	__cols__=0;
	__pitch__=0;
}
template <typename T,typename IT>
__attr_fiffhd__ uchar* MatrixHandler<T,IT>::operator *() const {
	return __data__;
}

template <typename T,typename IT> template <typename Allocator,uint alignment>
inline MatrixHandler<T,IT>& MatrixHandler<T,IT>::operator=(const Matrix<T,Allocator,alignment>& matrix) {
	__data__=*matrix;
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__pitch__=matrix.pitch();
	return *this;
}
template <typename T,typename IT> template <typename Allocator,uint alignment>
inline MatrixHandler<T,IT>& MatrixHandler<T,IT>::operator<<(const Matrix<T,Allocator,alignment>& matrix) {
	__data__=*matrix;
	__rows__=matrix.rows();
	__cols__=matrix.cols();
	__pitch__=matrix.pitch();
	return *this;
}

template <typename T,typename IT>
__attr_fiffhd__ T& MatrixHandler<T,IT>::operator()(IT i,IT j) {
	return *(reinterpret_cast<T*>((__data__)+__pitch__*i)+j);
}
template <typename T,typename IT>
__attr_fiffhd__ const T& MatrixHandler<T,IT>::operator()(IT i,IT j) const {
	return *(reinterpret_cast<T*>((__data__)+__pitch__*i)+j);
}
template <typename T,typename IT>
__attr_fiffhd__ T& MatrixHandler<T,IT>::operator[](IT i) {
	size_t r=i/__cols__;
	size_t c=i%__cols__;
	return *(reinterpret_cast<T*>((__data__)+__pitch__*r)+c);
}
template <typename T,typename IT>
__attr_fiffhd__ const T& MatrixHandler<T,IT>::operator[](IT i) const {
	size_t r=i/__cols__;
	size_t c=i%__cols__;
	return *(reinterpret_cast<T*>((__data__)+__pitch__*r)+c);
}

template <typename T,typename IT>
__attr_fiffhd__ IT MatrixHandler<T,IT>::n() const {
	return __rows__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixHandler<T,IT>::m() const {
	return __cols__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixHandler<T,IT>::rows() const {
	return __rows__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixHandler<T,IT>::cols() const {
	return __cols__;
}
template <typename T,typename IT>
__attr_fiffhd__ IT MatrixHandler<T,IT>::pitch() const {
	return __pitch__;
}

template <typename T,typename IT> template <ReadMode readMode>
__attr_fiffhd__ T MatrixHandler<T,IT>::read(IT i,IT j) const {
	return read_memory<readMode>(reinterpret_cast<T*>((__data__)+__pitch__*i)+j);
}
template <typename T,typename IT> template <ReadMode readMode>
__attr_fiffhd__ T MatrixHandler<T,IT>::read(IT i) const {
	size_t r=i/__cols__;
	size_t c=i%__cols__;
	return read_memory<readMode>(reinterpret_cast<T*>((__data__)+__pitch__*r)+c);
}
}
}
}
#endif
