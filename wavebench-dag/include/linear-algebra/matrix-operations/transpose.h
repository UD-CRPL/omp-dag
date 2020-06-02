#ifndef __TRANSPOSE_MATRIX_OPERATIONS_H__
#define __TRANSPOSE_MATRIX_OPERATIONS_H__

#include <vector>

#include "../matrix-formats/matrix-formats.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T,typename AllocatorT1,typename AllocatorT2,typename IT,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void transpose(MatrixCXS<T,IT,AllocatorT2,CRS>& tranpose,const MatrixCXS<T,IT,AllocatorT1,CRS>& matrix) {
	tranpose.resize(matrix.m(),matrix.n(),matrix.nzv());
	std::vector<IT> count(tranpose.rows());
	for(size_t i=0;i<tranpose.nzv();++i)
		count[matrix.indxs(i)]+=1;
	tranpose.ptr(0)=0;
	for(size_t i=0;i<tranpose.n();++i)
		tranpose.ptr(i+1)=count[i]+tranpose.ptr(i);
	for(size_t j=0;j<matrix.n();++j) {
		for(IT i=matrix.ptr(j);i<matrix.ptr(j+1);++i) {
			IT indx=matrix.indxs(i);
			IT ptrPos=tranpose.ptr(indx+1)-count[indx];
			count[indx]-=1;
			tranpose.indxs(ptrPos)=j;
			tranpose.values(ptrPos)=matrix.values(i);
		}
	}
}
template <typename T,typename AllocatorT1,typename AllocatorT2,typename IT,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void transpose(MatrixCXS<T,IT,AllocatorT2,CCS>& tranpose,const MatrixCXS<T,IT,AllocatorT1,CCS>& matrix) {
	tranpose.resize(matrix.m(),matrix.n(),matrix.nzv());
	std::vector<IT> count(tranpose.rows());
	for(size_t i=0;i<tranpose.nzv();++i)
		count[matrix.indxs(i)]+=1;
	tranpose.ptr(0)=0;
	for(size_t i=0;i<tranpose.m();++i)
		tranpose.ptr(i+1)=count[i]+tranpose.ptr(i);
	for(size_t j=0;j<matrix.m();++j) {
		for(IT i=matrix.ptr(j);i<matrix.ptr(j+1);++i) {
			IT indx=matrix.indxs(i);
			IT ptrPos=tranpose.ptr(indx+1)-count[indx];
			count[indx]-=1;
			tranpose.indxs(ptrPos)=j;
			tranpose.values(ptrPos)=matrix.values(i);
		}
	}
}
}
}
}
#endif
