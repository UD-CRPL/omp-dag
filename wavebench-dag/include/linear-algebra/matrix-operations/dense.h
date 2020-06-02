#ifndef __DENSE_MATRIX_OPERATIONS_H__
#define __DENSE_MATRIX_OPERATIONS_H__

#include <chrono>
#include <random>
#include "../matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T,typename AllocatorT1,typename AllocatorT2,uint Alignment1,uint Alignment2,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void transpose(Matrix<T,AllocatorT1,Alignment1>& dmatrix,const Matrix<T,AllocatorT2,Alignment2>& smatrix) {
	dmatrix.resize(smatrix.cols(),smatrix.rows(),0);
	dmatrix=0;
	for(size_t i=0;i<dmatrix.rows();++i)
		for(size_t j=0;j<dmatrix.cols();++j)
			dmatrix(i,j)=smatrix(j,i);
}
template <typename T,typename AllocatorT,uint Alignment,enable_IT<eq_CE(AllocatorT::location,HOST)> = 0>
Matrix<T,AllocatorT,Alignment> transpose(const Matrix<T,AllocatorT,Alignment>& matrix) {
	Matrix<T,AllocatorT,Alignment> result;
	transpose(result,matrix);
	return result;
}
template <typename T,typename AllocatorT1,typename AllocatorT2,typename AllocatorT3,uint Alignment1,uint Alignment2,uint Alignment3,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)&&eq_CE(AllocatorT3::location,HOST)> = 0>
void multiply(Matrix<T,AllocatorT1,Alignment1>& matrix,const Matrix<T,AllocatorT2,Alignment2>& A,const Matrix<T,AllocatorT3,Alignment3>& B) {
	error(A.cols()==B.rows(),"Invalid shapes for the matrices.",RUNTIME_ERROR,throw_error);
	matrix.resize(A.rows(),B.cols(),0);
	matrix=0;
	for(size_t i=0;i<matrix.rows();++i)
		for(size_t j=0;j<matrix.cols();++j)
			for(size_t k=0;k<A.cols();++k)
				matrix(i,j)+=A(i,k)*B(k,j);
}

template <typename T,typename AllocatorT=cpuAllocator,uint Alignment=128> Matrix<T,AllocatorT,Alignment> randomTriangularMatrix(size_t n,size_t m,T a=0.,T b=1.) {
	size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<double>distribution(-1.0,0.0);
	Matrix<T,AllocatorT,Alignment> A(n,m,-1,0);
	for(size_t i=0;i<n;++i)
		for(size_t j=0;j<m;++j)
			if(i<=j)
				A(i,j)=(-distribution(generator))*(b-a)+a;
	return A;
}
template <typename T,typename AllocatorT=cpuAllocator,uint Alignment=128> Matrix<T,AllocatorT,Alignment> randomTriangularMatrix(size_t n,T a=0.,T b=1.) {
    return randomTriangularMatrix<T,AllocatorT,Alignment>(n,n,a,b);
}
template <typename T,typename AllocatorT=cpuAllocator,uint Alignment=128> Matrix<T,AllocatorT,Alignment> randomPDMatrix(size_t n,T a=0.,T b=1.) {
	Matrix<T,AllocatorT,Alignment> B=randomTriangularMatrix<T,AllocatorT,Alignment>(n,a,b),result;
	auto BT=transpose(B);
	multiply(result,B,BT);
	return result;
}
template <typename T,typename AllocatorT1,typename AllocatorT2,uint Alignment1,uint Alignment2,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
T maxDifference(const Matrix<T,AllocatorT1,Alignment1>& A,const Matrix<T,AllocatorT2,Alignment2>& B) {
	error(A.cols()==B.cols()&&A.rows()==B.rows(),"Invalid shapes for the matrices.",RUNTIME_ERROR,throw_error);
	T result=0;
	for(size_t i=0;i<A.rows();++i)
		for(size_t j=0;j<A.cols();++j)
			result=__max__(result,__abs__(A(i,j)-B(i,j)));
	return result;
}
}
}
}
#endif
