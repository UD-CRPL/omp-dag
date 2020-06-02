#ifndef __SYMMETRIZE_MATRIX_OPERATIONS_H__
#define __SYMMETRIZE_MATRIX_OPERATIONS_H__

#include <vector>

#include "../matrix-formats/matrix-formats.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T,typename AllocatorT1,typename AllocatorT2,typename IT,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void symmetrize(MatrixCXS<T,IT,AllocatorT2,CRS>& S,const MatrixCXS<T,IT,AllocatorT1,CRS>& A) {
	S.resize(A.m(),A.n(),A.nzv());
	std::vector<IT> count(A.rows());
	for(size_t i=0;i<S.nzv();++i)
		count[A.indxs(i)]+=1;
//	S.ptr(0)=0;
//	for(size_t i=0;i<S.n();++i)
//		S.ptr(i+1)=count[i]+S.ptr(i);
//	for(size_t j=0;j<A.n();++j) {
//		for(IT i=A.ptr(j);i<A.ptr(j+1);++i) {
//			IT indx=A.indxs(i);
//			IT ptrPos=S.ptr(indx+1)-count[indx];
//			count[indx]-=1;
//			S.indxs(ptrPos)=j;
//			S.values(ptrPos)=A.values(i);
//		}
//	}
}
}
}
}

#endif
