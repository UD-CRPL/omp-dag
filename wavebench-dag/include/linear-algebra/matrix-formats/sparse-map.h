#ifndef __SPARSE_MAP_MATRIX_FORMATS_H__
#define __SPARSE_MAP_MATRIX_FORMATS_H__

#include <map>

#include "../../core/core.h"
#include "enum-definitions.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T> class MatrixMap {
public:
	typedef T type;
	typedef size_2A index_type;
private:
	std::map<size_2A,T> __data__;
	size_t __rows__=0;
	size_t __cols__=0;
public:
	MatrixMap();
	MatrixMap(size_t n);
	MatrixMap(size_t n,size_t m);
	MatrixMap(const MatrixMap& matrix);
	MatrixMap(MatrixMap&& matrix);
	~MatrixMap();

	MatrixMap& operator=(MatrixMap&& matrix);
	MatrixMap& operator=(const MatrixMap& matrix);

	std::map<size_2A,T>& operator*();
	const std::map<size_2A,T>& operator*() const;

	inline T& operator()(size_t i,size_t j);
	inline const T& operator()(size_t i,size_t j) const;
	inline T& operator[](size_2A ij);
	inline const T& operator[](size_2A ij) const;

	inline T& at(size_t i,size_t j);
	inline const T& at(size_t i,size_t j) const;
	inline T& at(index_type ij);
	inline const T& at(index_type ij) const;

	void clear();
	void reshape(size_t n);
	void reshape(size_t n,size_t m);

	void set(T v=0);

	inline size_t n() const;
	inline size_t m() const;
	inline size_t rows() const;
	inline size_t cols() const;
	inline size_t nzv() const;

	std::ostream& print(std::ostream &ost=std::cerr,std::size_t size=std::numeric_limits<size_t>::max(),std::string separator=",",	std::string begin="{",std::string end="}",std::function<void(std::ostream &,size_t,size_t,T)> pprinter = [](std::ostream &__ost__,size_t r,size_t c,T v) -> void { __ost__<<"["<<r<<","<<c<<"]->"<<v; }) const;
};
template<typename T>
inline MatrixMap<T>::MatrixMap(): __data__(std::map<size_2A,T>()) {
}
template<typename T>
inline MatrixMap<T>::MatrixMap(size_t n): __data__(std::map<size_2A,T>()),__rows__(n),__cols__(n) {
}
template<typename T>
inline MatrixMap<T>::MatrixMap(size_t n,size_t m): __data__(std::map<size_2A,T>()),__rows__(n),__cols__(m) {
}
template<typename T>
inline MatrixMap<T>::MatrixMap(const MatrixMap& matrix): __data__(matrix.__data__),__rows__(matrix.__rows__),__cols__(matrix.__cols__) {
}
template<typename T>
inline MatrixMap<T>::MatrixMap(MatrixMap&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
}
template<typename T>
inline MatrixMap<T>::~MatrixMap() {
	__data__.clear();
	__rows__=0;
	__cols__=0;
}

template<typename T>
inline MatrixMap<T>& MatrixMap<T>::operator=(MatrixMap&& matrix) {
	__data__=std::move(matrix.__data__);
	__memory__::move(__rows__,matrix.__rows__);
	__memory__::move(__cols__,matrix.__cols__);
	return *this;
}
template<typename T>
inline MatrixMap<T>& MatrixMap<T>::operator =(const MatrixMap& matrix) {
	__data__=matrix.__data__;
	__rows__=matrix.__rows__;
	__cols__=matrix.__cols__;
	return *this;
}

template<typename T>
inline std::map<size_2A,T>& MatrixMap<T>::operator *()  {
	return __data__;
}
template<typename T>
inline const std::map<size_2A,T>& MatrixMap<T>::operator *() const {
	return __data__;
}

template<typename T>
inline T& MatrixMap<T>::operator ()(size_t i,size_t j) {
	return __data__[size_2A({i,j})];
}
template<typename T>
inline const T& MatrixMap<T>::operator ()(size_t i,size_t j) const {
	return __data__[size_2A({i,j})];
}
template<typename T>
inline T& MatrixMap<T>::operator [](size_2A ij) {
	return __data__[ij];
}
template<typename T>
inline const T& MatrixMap<T>::operator [](size_2A ij) const {
	return __data__[ij];
}

template<typename T>
inline T& MatrixMap<T>::at(size_t i,size_t j) {
	return __data__.at(size_2A({i,j}));
}
template<typename T>
inline const T& MatrixMap<T>::at(size_t i,size_t j) const {
	return __data__.at(size_2A({i,j}));
}
template<typename T>
inline T& MatrixMap<T>::at(size_2A ij) {
	return __data__.at(ij);
}
template<typename T>
inline const T& MatrixMap<T>::at(size_2A ij) const {
	return __data__.at(ij);
}

template<typename T>
inline void MatrixMap<T>::clear() {
	__data__.clear();
	__rows__=0;
	__cols__=0;
}

template<typename T>
inline size_t MatrixMap<T>::n() const {
	return __rows__;
}
template<typename T>
inline size_t MatrixMap<T>::m() const {
	return __cols__;
}
template<typename T>
inline size_t MatrixMap<T>::rows() const {
	return __rows__;
}
template<typename T>
inline size_t MatrixMap<T>::cols() const {
	return __cols__;
}
template<typename T>
inline size_t MatrixMap<T>::nzv() const {
	return __data__.size();
}
template<typename T>
inline void MatrixMap<T>::reshape(size_t i) {
	__rows__=i;
	__cols__=i;
}
template<typename T>
inline void MatrixMap<T>::reshape(size_t i,size_t j) {
	__rows__=i;
	__cols__=j;
}

template<typename T>
inline void MatrixMap<T>::set(T v) {
	for(auto it=__data__.begin();it!=__data__.end();++it)
		it->second=v;
}

template<typename T>
inline std::ostream& MatrixMap<T>::print(std::ostream& ost,std::size_t size,std::string separator,std::string begin,std::string end,std::function<void(std::ostream&,size_t,size_t,T)> pprinter) const {
	ost<<begin;
	size_t c=0;
	for(auto i=__data__.cbegin();i!=__data__.cend()&&(c<size);++i) {
		pprinter(ost,i->first[0],i->first[1],i->second);
		if(c<(__data__.size()-1))
			ost<<separator;
		++c;
	}
	ost<<end;
    return ost;
}
template<typename T>
std::ostream& operator<<(std::ostream& ost,const MatrixMap<T>& matrix) {
	return matrix.print(ost);
}
}
}
}
#endif
