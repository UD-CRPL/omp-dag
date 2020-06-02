#ifndef __RBIO_SPARSE_IO_H__
#define __RBIO_SPARSE_IO_H__

#include <suitesparse/RBio.h>

#include "../../linear-algebra/linear-algebra.h"

namespace __third_party__ {
namespace __sparse_io__ {
namespace __rbio__ {
struct RBioInfo {
	SuiteSparse_long build_upper=1;
	SuiteSparse_long zero_handling=1;
	char title [73]={0};
	char key [9]={0};
    char mtype [4]={0};
    SuiteSparse_long nrow=0;
    SuiteSparse_long ncol=0;
    SuiteSparse_long mkind=0;
    SuiteSparse_long skind=0;
    SuiteSparse_long asize=0;
    SuiteSparse_long znz=0;
    SuiteSparse_long *Ap=nullptr;
    SuiteSparse_long *Ai=nullptr;
	double *Ax=nullptr;
	double *Az=nullptr;
	SuiteSparse_long *Zp=nullptr;
	SuiteSparse_long *Zi=nullptr;
	void free() {
		__core__::free_memory(Ap);
		__core__::free_memory(Ai);
		__core__::free_memory(Ax);
		__core__::free_memory(Az);
		__core__::free_memory(Zp);
		__core__::free_memory(Zi);
	}
};
template <typename T,typename IT> __core__::MatrixCXS<T,IT,__core__::cpuAllocator,__core__::CCS> read(std::string filename,int buildUpper=1,int zeroHandling=1) {
	using namespace __core__;
	MatrixCXS<T,IT,cpuAllocator,CCS> result;
	RBioInfo info;
	int ecode=RBread(const_cast<char*>(filename.c_str()),buildUpper,zeroHandling,info.title,info.key,info.mtype,&(info.nrow),&(info.ncol),&(info.mkind),&(info.skind),&(info.asize),
			&(info.znz),&(info.Ap),&(info.Ai),&(info.Ax),NULL,NULL,NULL);
	if(ecode!=0) {
		error(false,"Error reading the matrix.",RUNTIME_ERROR,throw_error);
	}
	result.resize(info.nrow,info.ncol,info.Ap[info.ncol]);
	for(size_t i=0;i<=result.cols();++i)
		result.ptr(i)=info.Ap[i];
	for(size_t i=0;i<result.nzv();++i) {
		result.indxs(i)=info.Ai[i];
		result.values(i)=info.Ax[i];
	}
	info.free();
	return result;
}
}
}
}
#endif
