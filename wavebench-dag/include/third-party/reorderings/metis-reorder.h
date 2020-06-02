#ifndef __METIS_REORDER_H__
#define __METIS_REORDER_H__

#include <vector>

#include <metis.h>

#include "../../linear-algebra/linear-algebra.h"

namespace __third_party__ {
namespace __reordeings__ {
namespace __metis__ {
template <typename T,typename IT,typename Allocator,__core__::enable_IT<sizeof(IT)==4> =0>
int reorderPermuation(const __core__::MatrixCXS<T,IT,Allocator,__core__::CCS>& A,std::vector<idx_t>& permutation,std::vector<idx_t>& invPermutation) {
	idx_t n=A.m();
	permutation.resize(n);
	invPermutation.resize(n);
	idx_t options[METIS_NOPTIONS]={0};
	METIS_SetDefaultOptions(options);
	options[METIS_OPTION_NUMBERING]=0;
	return METIS_NodeND(&n,A.ptr(),A.indxs(),NULL,options,permutation.data(),invPermutation.data());
}
}
}
}
#endif
