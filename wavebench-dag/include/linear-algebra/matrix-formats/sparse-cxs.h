#ifndef __SPARSE_CXS_MATRIX_FORMATS_H__
#define __SPARSE_CXS_MATRIX_FORMATS_H__

#include "../../core/core.h"
#include "enum-definitions.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> class MatrixCXS {
public:
	typedef T type;
	typedef Allocator allocator_type;
	typedef typename Allocator::memory_type memory_type;
	static constexpr SparseCXSFormat format=frmt;
private:
	Array<uchar,Allocator> __data__;
	size_t __rows__=0;
	size_t __cols__=0;
	size_t __nzv__=0;
	IT* __ptr__=nullptr;
	IT* __indx__=nullptr;
	T*  __values__=nullptr;
	template <typename TT,typename ITT,typename allocatorType_T,SparseCXSFormat frmtT> friend class MatrixCXS;
	static size_t computeSize(size_t n,size_t m,size_t nzv);
	static size_t computeXSize(size_t n,size_t m,size_t nzv);
	static size_t computeYSize(size_t n,size_t m,size_t nzv);
	static size_t computeVSize(size_t n,size_t m,size_t nzv);
	static size_t computeYPos(size_t n,size_t m,size_t nzv);
	static size_t computeVPos(size_t n,size_t m,size_t nzv);
	void computePointers(size_t n,size_t m,size_t nzv);
public:
	MatrixCXS(int dev=-1);
	MatrixCXS(const MatrixCXS& matrix);
	MatrixCXS(MatrixCXS&& matrix);
	MatrixCXS(size_t n,size_t nzv,int dev,StreamType stream=0,const Allocator& allocator=Allocator());
	MatrixCXS(size_t n,size_t m,size_t nzv,int dev,StreamType stream=0,const Allocator& allocator=Allocator());
	template <typename allocatorType_T> MatrixCXS(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix,int dev=-1,StreamType stream=0,const Allocator& allocator=Allocator());
	~MatrixCXS();

	MatrixCXS& operator=(MatrixCXS&& matrix);
	MatrixCXS& operator=(const MatrixCXS& matrix);
	template <typename allocatorType_T> MatrixCXS& operator=(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix);

	uchar* operator*() const;

	int device() const;

	IT* ptr() const;
	IT* indxs() const;
	T* values() const;
	IT& ptr(size_t i);
	const IT& ptr(size_t i) const;
	IT& indxs(size_t i);
	const IT& indxs(size_t i) const;
	T& values(size_t i);
	const T& values(size_t i) const;

	void free();
	void clear();

	inline size_t n() const;
	inline size_t m() const;
	inline size_t rows() const;
	inline size_t cols() const;
	inline size_t nzv() const;

	void resize(size_t i,size_t nzv,StreamType stream=0);
	void resize(size_t i,size_t j,size_t nzv,StreamType stream=0);
	void set(int v=0,StreamType stream=0);

	void setNZV(size_t nzv);

	template <bool check=true,typename allocatorType_T=void> MatrixCXS& import(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix,StreamType stream=0);

	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0> std::ostream& print(std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=", ",
			std::string begin="{",std::string end="}",std::function<void(std::ostream &,IT,IT,T)> pprinter = [](std::ostream &__ost__,IT r,IT c,T v) -> void { __ost__<<"["<<r<<","<<c<<"]->"<<v; }) const;
};
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeSize(size_t n,size_t m,size_t nzv) {
	return computeXSize(n,m,nzv)+computeYSize(n,m,nzv)+computeVSize(n,m,nzv);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeXSize(size_t n,size_t m,size_t nzv) {
	constexpr size_t alignment=256;
	size_t tmp;
	switch(frmt) {
	case COO:
		tmp=sizeof(IT)*nzv;
		break;
	case CRS:
		tmp=sizeof(IT)*(n+1);
		break;
	case CCS:
		tmp=sizeof(IT)*(m+1);
		break;
	}
	return alignment*((tmp+alignment-1)/alignment);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeYSize(size_t n,size_t m,size_t nzv) {
	constexpr size_t alignment=256;
	size_t tmp=sizeof(IT)*nzv;
	return alignment*((tmp+alignment-1)/alignment);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeVSize(size_t n,size_t m,size_t nzv) {
	constexpr size_t alignment=256;
	size_t tmp=sizeof(T)*nzv;
	return alignment*((tmp+alignment-1)/alignment);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeYPos(size_t n,size_t m,size_t nzv) {
	return computeXSize(n,m,nzv);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::computeVPos(size_t n,size_t m,size_t nzv) {
	return computeXSize(n,m,nzv)+computeYSize(n,m,nzv);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::computePointers(size_t n,size_t m,size_t nzv) {
	__ptr__=reinterpret_cast<IT*>(*__data__);
	__indx__=reinterpret_cast<IT*>((*__data__)+computeYPos(n,m,nzv));
	__values__=reinterpret_cast<T*>((*__data__)+computeVPos(n,m,nzv));
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(int dev):__data__(Array<uchar,Allocator>(dev)) {
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(const MatrixCXS& matrix): __data__(matrix.__data__),__rows__(matrix.__rows__),__cols__(matrix.__cols__),__nzv__(matrix.__nzv__) {
	computePointers(__rows__,__cols__,__nzv__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(MatrixCXS&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
	__memory__::move(__nzv__,matrix.__nzv__);
	__memory__::move(__ptr__,matrix.__ptr__);
	__memory__::move(__indx__,matrix.__indx__);
	__memory__::move(__values__,matrix.__values__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(size_t n,size_t nzv,int dev,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(computeSize(n,n,nzv),dev,0,stream,allocator)),__rows__(n),__cols__(n),__nzv__(nzv) {
	computePointers(__rows__,__cols__,__nzv__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(size_t n,size_t m,size_t nzv,int dev,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(computeSize(n,m,nzv),dev,0,stream,allocator)),__rows__(n),__cols__(m),__nzv__(nzv)  {
	computePointers(__rows__,__cols__,__nzv__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> template<typename allocatorType_T>
inline MatrixCXS<T,IT,Allocator,frmt>::MatrixCXS(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix,int dev,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(matrix.__data__,dev,stream,allocator)),__rows__(matrix.__rows__),__cols__(matrix.__cols__),__nzv__(matrix.__nzv__) {
	computePointers(__rows__,__cols__,__nzv__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>::~MatrixCXS() {
	__data__.free();
	__data__.free_allocator();
	__rows__=0;
	__cols__=0;
	__nzv__=0;
	__ptr__=nullptr;
	__indx__=nullptr;
	__values__=nullptr;
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>& MatrixCXS<T,IT,Allocator,frmt>::operator=(MatrixCXS&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
	__memory__::move(__nzv__,matrix.__nzv__);
	__memory__::move(__ptr__,matrix.__ptr__);
	__memory__::move(__indx__,matrix.__indx__);
	__memory__::move(__values__,matrix.__values__);
	return *this;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline MatrixCXS<T,IT,Allocator,frmt>& MatrixCXS<T,IT,Allocator,frmt>::operator=(const MatrixCXS& matrix) {
	return import<true>(matrix,DEFAULT_STREAM);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> template<typename allocatorType_T>
inline MatrixCXS<T,IT,Allocator,frmt>& MatrixCXS<T,IT,Allocator,frmt>::operator=(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix) {
	return import<true>(matrix,DEFAULT_STREAM);
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline uchar* MatrixCXS<T,IT,Allocator,frmt>::operator*() const {
	return *__data__;
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline int MatrixCXS<T,IT,Allocator,frmt>::device() const {
	return __data__.device();
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline IT* MatrixCXS<T,IT,Allocator,frmt>::ptr() const {
	return __ptr__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline IT* MatrixCXS<T,IT,Allocator,frmt>::indxs() const {
	return __indx__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline T* MatrixCXS<T,IT,Allocator,frmt>::values() const {
	return __values__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline IT& MatrixCXS<T,IT,Allocator,frmt>::ptr(size_t i) {
	return __ptr__[i];
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline const IT& MatrixCXS<T,IT,Allocator,frmt>::ptr(size_t i) const {
	return __ptr__[i];
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline IT& MatrixCXS<T,IT,Allocator,frmt>::indxs(size_t i) {
	return __indx__[i];
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline const IT& MatrixCXS<T,IT,Allocator,frmt>::indxs(size_t i) const {
	return __indx__[i];
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline T& MatrixCXS<T,IT,Allocator,frmt>::values(size_t i) {
	return __values__[i];
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline const T& MatrixCXS<T,IT,Allocator,frmt>::values(size_t i) const {
	return __values__[i];
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::free() {
	__rows__=0;
	__cols__=0;
	__nzv__=0;
	__data__.free();
	__ptr__=nullptr;
	__indx__=nullptr;
	__values__=nullptr;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::clear() {
	__rows__=0;
	__cols__=0;
	__nzv__=0;
	__data__.clear();
	computePointers(__rows__,__cols__,__nzv__);
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::set(int v,StreamType stream) {
	__data__.set(v,stream);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::setNZV(size_t nzv) {
	if(nzv<=__nzv__)
		__nzv__=nzv;
	else {
		error(true,"Invalid value.",RUNTIME_ERROR,stderr_error);
	}
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::n() const {
	return __rows__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::m() const {
	return __cols__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::rows() const {
	return __rows__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::cols() const {
	return __cols__;
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline size_t MatrixCXS<T,IT,Allocator,frmt>::nzv() const {
	return __nzv__;
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::resize(size_t i,size_t nzv,StreamType stream) {
	if(computeSize(i,i,nzv)>__data__.capacity())
		__data__.reserve(computeSize(i,i,nzv),stream);
	__data__.expand();
	__rows__=i;
	__cols__=i;
	__nzv__=nzv;
	computePointers(__rows__,__cols__,__nzv__);
}
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
inline void MatrixCXS<T,IT,Allocator,frmt>::resize(size_t i,size_t j,size_t nzv,StreamType stream) {
	if(computeSize(i,j,nzv)>__data__.capacity())
		__data__.reserve(computeSize(i,j,nzv),stream);
	__data__.expand();
	__rows__=i;
	__cols__=j;
	__nzv__=nzv;
	computePointers(__rows__,__cols__,__nzv__);
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> template <bool check,typename allocatorType_T>
MatrixCXS<T,IT,Allocator,frmt>& MatrixCXS<T,IT,Allocator,frmt>::import(const MatrixCXS<T,IT,allocatorType_T,frmt>& matrix,StreamType stream) {
	__data__.template import<check>(matrix.__data__,stream);
	__rows__=matrix.__rows__;
	__cols__=matrix.__cols__;
	__nzv__=matrix.__nzv__;
	computePointers(__rows__,__cols__,__nzv__);
	return *this;
}

template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> template<typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
inline std::ostream& MatrixCXS<T,IT,Allocator,frmt>::print(std::ostream &ost,std::size_t size,std::string separator,
		std::string begin,std::string end,std::function<void(std::ostream &,IT,IT,T)> pprinter ) const {
	ost<<begin;
	size_t counter=0;
	switch(frmt) {
	case COO:
		for(size_t i=0;(i<__nzv__)&&(counter<size);++i) {
			pprinter(ost,__ptr__[i],__indx__[i],__values__[i]);
			if(counter<(__nzv__-1))
				ost<<separator;
			counter++;
		}
		break;
	case CRS:
		for(size_t i=0;(i<__rows__)&&(counter<size);++i) {
			for(size_t j=__ptr__[i];j<__ptr__[i+1];++j) {
				pprinter(ost,i,__indx__[j],__values__[j]);
				if(counter<(__nzv__-1))
					ost<<separator;
				counter++;
			}
		}
		break;
	case CCS:
		for(size_t i=0;(i<__cols__)&&(counter<size);++i) {
			for(size_t j=__ptr__[i];j<__ptr__[i+1];++j) {
				pprinter(ost,__indx__[j],i,__values__[j]);
				if(counter<(__nzv__-1))
					ost<<separator;
				counter++;
			}
		}
		break;
	}
	ost<<end;
    return ost;
}
template<typename T,typename IT,typename Allocator,SparseCXSFormat frmt,enable_IT<eq_CE(Allocator::location,HOST)> = 0>
std::ostream& operator<<(std::ostream& ost,const MatrixCXS<T,IT,Allocator,frmt>& matrix) {
	return matrix.print(ost);
}
}
}
}
#endif
