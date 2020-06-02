#ifndef __DENSE_MATRIX_MATRIX_FORMATS_H__
#define __DENSE_MATRIX_MATRIX_FORMATS_H__

#include "../../core/core.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T,typename Allocator,uint ALIGNMENT=128> class Matrix {
public:
	typedef T type;
	typedef Allocator allocator_type;
	typedef typename Allocator::memory_type memory_type;
	static constexpr uint alignment=ALIGNMENT;
	static_assert(is_pow2_CE(alignment),"Invalid alignment.");
	struct __assign__;
private:
	Array<uchar,Allocator> __data__;
	size_t __rows__=0;
	size_t __cols__=0;
	size_t __pitch__=0;
	template <typename TT,typename allocatorType_T,uint algnmnt> friend class Matrix;
	static size_t computePitch(size_t rows,size_t cols);
	static size_t computeSize(size_t rows,size_t cols);
public:
	Matrix(int dev=-1);
	Matrix(const Matrix& matrix);
	Matrix(Matrix&& matrix);
	Matrix(size_t n,int dev,T v,StreamType stream=0,const Allocator& allocator=Allocator());
	Matrix(size_t n,size_t m,int dev,T v,StreamType stream=0,const Allocator& allocator=Allocator());
	template <typename allocatorType_T> Matrix(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix,int dev=-1,StreamType stream=0,const Allocator& allocator=Allocator());
	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0> Matrix(std::initializer_list<T> list,size_t n,size_t m,const allocatorType_T& allocator=allocatorType_T());
	~Matrix();

	Matrix& operator=(T v);
	Matrix& operator=(Matrix&& matrix);
	Matrix& operator=(const Matrix& matrix);
	Matrix& operator=(std::initializer_list<T> list);
	template <typename allocatorType_T> Matrix& operator=(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix);

	uchar* operator*() const;

	inline T& operator()(size_t i,size_t j);
	inline const T& operator()(size_t i,size_t j) const;
	inline T& operator[](size_t i);
	inline const T& operator[](size_t i) const;

	inline T& at(size_t i,size_t j);
	inline const T& at(size_t i,size_t j) const;
	inline T& at(size_t i);
	inline const T& at(size_t i) const;

	void free();
	void clear();

	inline size_t n() const;
	inline size_t m() const;
	inline size_t rows() const;
	inline size_t cols() const;
	inline size_t pitch() const;

	void set(T val=0,StreamType stream=0);

	void resize(size_t i,T val=0,StreamType stream=0);
	void resize(size_t i,size_t j,T val=0,StreamType stream=0);
	void reshape(size_t i);
	void reshape(size_t i,size_t j);

	template <bool check=true,typename allocatorType_T=void> Matrix& import(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix,StreamType stream=0);

	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0> std::ostream& print(std::ostream &ost=std::cerr,std::string separator="\t",
			std::string rbegin="",std::string rend="",std::string begin="",std::string end="",std::string newRow="\n") const;
};
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::computePitch(size_t rows,size_t cols) {
	size_t colsBITS=cols*sizeof(T);
	if(ALIGNMENT!=0)
		return ALIGNMENT*((colsBITS+ALIGNMENT-1)/ALIGNMENT);
	else
		return colsBITS;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::computeSize(size_t rows,size_t cols) {
	return rows*computePitch(rows,cols);
}
template<typename T,typename Allocator,uint ALIGNMENT> struct Matrix<T,Allocator,ALIGNMENT>::__assign__ {
	template <typename V=T> static __host_device__ __optimize__ __forceinline__ void fn(size_t idx,size_t size,uchar* data,size_t rows,size_t cols,size_t pitch,V val){
		size_t i=idx/cols;
		size_t j=idx%cols;
		*(reinterpret_cast<T*>(data+pitch*i)+j)=val;
	}
};

template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(int dev): __data__(Array<uchar,Allocator>(dev)) {
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(const Matrix& matrix): __data__(matrix.__data__),__rows__(matrix.__rows__),__cols__(matrix.__cols__),__pitch__(matrix.__pitch__) {
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(Matrix&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
	__memory__::move(__pitch__,matrix.__pitch__);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(size_t n,int dev,T v,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(computeSize(n,n),dev,0,stream,allocator)),__rows__(n),__cols__(n),__pitch__(computePitch(n,n)) {
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<8,128,4,false>>(__rows__*__cols__,__data__.device(),stream,*__data__,__rows__,__cols__,__pitch__,v);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(size_t n,size_t m,int dev,T v,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(computeSize(n,m),dev,0,stream,allocator)),__rows__(n),__cols__(m),__pitch__(computePitch(n,m)) {
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<8,128,4,false>>(__rows__*__cols__,__data__.device(),stream,*__data__,__rows__,__cols__,__pitch__,v);
}
template<typename T,typename Allocator,uint ALIGNMENT> template<typename allocatorType_T>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix,int dev,StreamType stream,const Allocator& allocator):
	__data__(Array<uchar,Allocator>(matrix.__data__,dev,stream,allocator)),__rows__(matrix.__rows__),__cols__(matrix.__cols__),__pitch__(matrix.__pitch__) {
}
template<typename T,typename Allocator,uint ALIGNMENT> template<typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
inline Matrix<T,Allocator,ALIGNMENT>::Matrix(std::initializer_list<T> list,size_t n,size_t m,const allocatorType_T& allocator):
	__data__(Array<uchar,Allocator>(computeSize(n,m),-1,allocator)),__rows__(n),__cols__(m),__pitch__(computePitch(n,m)) {
	size_t i=0;
	for(auto it=list.begin();it!=list.end();++it)
		(*this)[i++]=*it;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>::~Matrix() {
	__data__.free();
	__data__.free_allocator();
	__rows__=0;
	__cols__=0;
	__pitch__=0;
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::operator=(T v) {
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<8,128,4,false>>(__rows__*__cols__,__data__.device(),DEFAULT_STREAM,*__data__,__rows__,__cols__,__pitch__,v);
	return *this;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::operator=(Matrix&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
	__memory__::move(__pitch__,matrix.__pitch__);
	return *this;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::operator=(const Matrix& matrix) {
	return import<true>(matrix,DEFAULT_STREAM);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::operator=(std::initializer_list<T> list) {
	size_t i=0;
	for(auto it=list.begin();it!=list.end()&&(i<__rows__*__cols__);++it)
		(*this)[i++]=*it;
}
template<typename T,typename Allocator,uint ALIGNMENT> template<typename allocatorType_T>
inline Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::operator=(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix) {
	return import<true>(matrix,DEFAULT_STREAM);
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline uchar* Matrix<T,Allocator,ALIGNMENT>::operator*() const {
	return *__data__;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline T& Matrix<T,Allocator,ALIGNMENT>::operator()(size_t i,size_t j) {
	return *(reinterpret_cast<T*>((*__data__)+__pitch__*i)+j);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline const T& Matrix<T,Allocator,ALIGNMENT>::operator()(size_t i,size_t j) const {
	return *(reinterpret_cast<T*>((*__data__)+__pitch__*i)+j);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline T& Matrix<T,Allocator,ALIGNMENT>::operator[](size_t i) {
	size_t r=i/__cols__;
	size_t c=i%__cols__;
	return *(reinterpret_cast<T*>((*__data__)+__pitch__*r)+c);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline const T& Matrix<T,Allocator,ALIGNMENT>::operator[](size_t i) const {
	size_t r=i/__cols__;
	size_t c=i%__cols__;
	return *(reinterpret_cast<T*>((*__data__)+__pitch__*r)+c);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline T& Matrix<T,Allocator,ALIGNMENT>::at(size_t i,size_t j) {
	if(i<__rows__&&j<__cols__)
		return (*this)(i,j);
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline const T& Matrix<T,Allocator,ALIGNMENT>::at(size_t i,size_t j) const {
	if(i<__rows__&&j<__cols__)
		return (*this)(i,j);
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline T& Matrix<T,Allocator,ALIGNMENT>::at(size_t i) {
	if(i<__rows__*__cols__)
		return (*this)[i];
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline const T& Matrix<T,Allocator,ALIGNMENT>::at(size_t i) const {
	if(i<__rows__*__cols__)
		return (*this)[i];
	else {
		error(true,"Index is out of bounds.",RUNTIME_ERROR,throw_error);
	}
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::free() {
	__rows__=0;
	__cols__=0;
	__pitch__=0;
	__data__.free();
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::clear() {
	__rows__=0;
	__cols__=0;
	__pitch__=0;
	__data__.clear();
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::n() const {
	return __rows__;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::m() const {
	return __cols__;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::rows() const {
	return __rows__;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::cols() const {
	return __cols__;
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline size_t Matrix<T,Allocator,ALIGNMENT>::pitch() const {
	return __pitch__;
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::set(T val,StreamType stream) {
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<1,128,1,false>>(__rows__*__cols__,__data__.device(),stream,*__data__,__rows__,__cols__,__pitch__,val);
}

template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::resize(size_t i,T val,StreamType stream) {
	if(computeSize(i,i)>__data__.capacity())
		__data__.resize(computeSize(i,i),0,stream);
	__rows__=i;
	__cols__=i;
	__pitch__=computePitch(i,i);
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<8,128,4,false>>(__rows__*__cols__,__data__.device(),stream,*__data__,__rows__,__cols__,__pitch__,val);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::resize(size_t i,size_t j,T val,StreamType stream) {
	if(computeSize(i,j)>__data__.capacity())
		__data__.resize(computeSize(i,j),0,stream);
	__rows__=i;
	__cols__=j;
	__pitch__=computePitch(i,j);
	apply_meta<__assign__,Allocator::location,ApplyArrayExecutionPolicy<8,128,1,false>>(__rows__*__cols__,__data__.device(),stream,*__data__,__rows__,__cols__,__pitch__,val);
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::reshape(size_t i) {
	if(computeSize(i,i)>__data__.capacity()) {
		error(true,"Invalid shape.",RUNTIME_ERROR,throw_error);
	}
	else {
		__rows__=i;
		__cols__=i;
		__pitch__=computePitch(i,i);
	}
}
template<typename T,typename Allocator,uint ALIGNMENT>
inline void Matrix<T,Allocator,ALIGNMENT>::reshape(size_t i,size_t j) {
	if(computeSize(i,j)>__data__.capacity()) {
		error(true,"Invalid shape.",RUNTIME_ERROR,throw_error);
	}
	else {
		__rows__=i;
		__cols__=j;
		__pitch__=computePitch(i,j);
	}
}

template<typename T,typename Allocator,uint ALIGNMENT> template <bool check,typename allocatorType_T>
Matrix<T,Allocator,ALIGNMENT>& Matrix<T,Allocator,ALIGNMENT>::import(const Matrix<T,allocatorType_T,ALIGNMENT>& matrix,StreamType stream) {
	__data__.template import<check>(matrix.__data__,stream);
	__rows__=matrix.__rows__;
	__cols__=matrix.__cols__;
	__pitch__=matrix.__pitch__;
	return *this;
}

template<typename T,typename Allocator,uint ALIGNMENT> template<typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
inline std::ostream& Matrix<T,Allocator,ALIGNMENT>::print(std::ostream &ost,std::string separator,std::string rbegin,std::string rend,std::string begin,std::string end,std::string newRow) const {
	ost<<begin;
    for(std::size_t i=0;i<__rows__;++i) {
    	ost<<rbegin;
        for(std::size_t j=0;j<__cols__;++j) {
        	ost<<(*this)(i,j);
            if(j!=__cols__-1)
            	ost<<separator;
        }
        if(i!=__rows__-1)
        	ost<<rend<<newRow;
        else
            ost<<rend<<end;
    }
    return ost;
}
template<typename T,typename Allocator,uint ALIGNMENT,enable_IT<eq_CE(Allocator::location,HOST)> = 0>
std::ostream& operator<<(std::ostream& ost,const Matrix<T,Allocator,ALIGNMENT>& matrix) {
	return matrix.print(ost);
}
}
}
}
#endif
