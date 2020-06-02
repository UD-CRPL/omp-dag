#ifndef __SERIAL_SMITH_WATERMAN_WAVEFRONT_H__
#define __SERIAL_SMITH_WATERMAN_WAVEFRONT_H__

#include "../../linear-algebra/matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
namespace __private__ {
template <typename FT=void,typename T=void,typename ST=void>
void smithWaterman(ArrayHandler<ST,size_t>& A,ArrayHandler<ST,size_t>& B,MatrixHandler<T,size_t>& M,FT& score,T gap) {
	for(size_t i=0;i<M.n();++i)
		M(i,0)=0;
	for(size_t i=0;i<M.m();++i)
		M(0,i)=0;
	for(size_t i=1;i<M.n();++i)
		for(size_t j=1;j<M.m();++j)
			M(i,j)=__max__(0,__max__(M(i-1,j-1)+score(A[i-1],B[j-1]),__max__(M(i-1,j)-gap,M(i,j-1)-gap)));
}
}
template <typename FT=void,typename T=void,typename Allocator=void,uint alignment=1>
double smithWaterman(std::string& A,std::string& B,Matrix<T,Allocator,alignment>& M,FT& score,T gap) {
	ArrayHandler<char,size_t>  AH,BH;
	AH.__data__=const_cast<char*>(A.c_str());
	AH.__size__=A.size();
	BH.__data__=const_cast<char*>(B.c_str());
	BH.__size__=B.size();
	M.resize(A.size()+1,B.size()+1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	cpu_timer timer;
	timer.start();
	__private__::smithWaterman(AH,BH,MH,score,gap);
	timer.stop();
	return timer.elapsed_time();
}
template <typename FT=void,typename T=void,typename IT=int,typename Allocator=void,uint alignment=1>
double smithWaterman(const Array<IT,Allocator>& A,const Array<IT,Allocator>& B,Matrix<T,Allocator,alignment>& M,FT& score,T gap) {
	ArrayHandler<IT,size_t>  AH(A),BH(B);
	M.resize(A.size()+1,B.size()+1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	cpu_timer timer;
	timer.start();
	__private__::smithWaterman(AH,BH,MH,score,gap);
	timer.stop();
	return timer.elapsed_time();
}
template <typename FT=void,typename T=void>
double smithWaterman(std::string& A,std::string& B,FT& score,T gap) {
	ArrayHandler<char,size_t>  AH,BH;
	AH.__data__=const_cast<char*>(A.c_str());
	AH.__size__=A.size();
	BH.__data__=const_cast<char*>(B.c_str());
	BH.__size__=B.size();
	Matrix<T,cpuAllocator,1> M(A.size()+1,B.size()+1,-1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	cpu_timer timer;
	timer.start();
	__private__::smithWaterman(AH,BH,MH,score,gap);
	timer.stop();
	return timer.elapsed_time();
}
}
}
}
#endif
