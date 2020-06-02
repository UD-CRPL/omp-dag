#ifndef __NON_ZERO_PATTERN_MATRIX_OPERATIONS_H__
#define __NON_ZERO_PATTERN_MATRIX_OPERATIONS_H__

#include <set>
#include <vector>

#include "../matrix-formats/matrix-formats.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt>
MatrixCXS<IT,IT,Allocator,CCS> colPattern(const MatrixCXS<T,IT,Allocator,frmt>& matrix,const MatrixCXS<IT,IT,Allocator,CRS>& etree) {
	std::vector<std::set<IT>> L(matrix.n());
	std::vector<IT> w(matrix.n(),-1);
	size_t nzv=0;
	for(IT k=0;k<matrix.n();++k) {
		L[k].insert(k);
		w[k]=k;
		for(IT p=matrix.ptr(k);p<matrix.ptr(k+1);++p) {
			IT j=matrix.indxs(p);
			if(k<=j)
				break;
			while(w[j]!=k) {
				L[j].insert(k);
				w[j]=k;
				j=etree.indxs(etree.ptr(j));
			}
		}
	}
	for(size_t k=0;k<matrix.n();++k)
		nzv+=L[k].size();
	MatrixCXS<IT,IT,Allocator,CCS> pattern(matrix.n(),nzv,-1);
	pattern.ptr(0)=0;
	IT pos=0;
	for(size_t k=0;k<matrix.n();++k) {
		for(auto it=L[k].begin();it!=L[k].end();++it) {
			pattern.indxs(pos)=(*it);
			pattern.values(pos++)=pos;
		}
		pattern.ptr(k+1)=pos;
	}
	return pattern;
}
template <typename T,typename IT,typename Allocator>
MatrixCXS<IT,IT,Allocator,CRS> rowPattern(const MatrixCXS<T,IT,Allocator,CCS>& matrix) {
	MatrixCXS<IT,IT,Allocator,CRS> pattern(matrix.n(),matrix.m(),matrix.nzv(),-1,DEFAULT_STREAM);
	std::vector<size_t> p(matrix.n(),0);
	pattern.ptr(0)=0;
	for(size_t k=0;k<matrix.nzv();++k)
		pattern.ptr(matrix.indxs(k)+1)+=1;
	for(size_t k=0;k<matrix.n();++k)
		pattern.ptr(k+1)+=pattern.ptr(k);
	for(size_t k=0;k<matrix.m();++k) {
		for(IT it=matrix.ptr(k);it<matrix.ptr(k+1);++it) {
			IT row=matrix.indxs(it);
			pattern.indxs(p[row]+pattern.ptr(row))=k;
			pattern.values(p[row]+pattern.ptr(row))=matrix.values(it);
			p[row]+=1;
		}
	}
	return pattern;
}
template <typename T,typename IT,typename AllocatorD,typename AllocatorS>
void copyPattern(MatrixCXS<T,IT,AllocatorD,CCS>& matrix,const MatrixCXS<IT,IT,AllocatorS,CCS>& pattern,StreamType stream=DEFAULT_STREAM) {
	matrix.resize(pattern.n(),pattern.m(),pattern.nzv(),stream);
	matrix.set(0,stream);
	AllocatorD::template copy<typename AllocatorD::memory_type,typename AllocatorS::memory_type>(&(matrix.ptr(0)),&(const_cast<IT&>(pattern.ptr(0))),matrix.m()+1,matrix.device(),pattern.device(),stream);
	AllocatorD::template copy<typename AllocatorD::memory_type,typename AllocatorS::memory_type>(&(matrix.indxs(0)),&(const_cast<IT&>(pattern.indxs(0))),matrix.nzv(),matrix.device(),pattern.device(),stream);
}
template <typename T,typename IT,typename AllocatorD,typename AllocatorS>
void copyValues(MatrixCXS<T,IT,AllocatorD,CCS>& matrix,const MatrixCXS<T,IT,AllocatorS,CCS>& A) {
	for(size_t k=0;k<A.m();++k) {
		IT pos=matrix.ptr(k);
		for(IT it=A.ptr(k);it<A.ptr(k+1);++it) {
			IT r=A.indxs(it);
			if(k<=r) {
				for(;pos<matrix.ptr(k+1);++pos) {
					if(matrix.indxs(pos)==r) {
						matrix.values(pos++)=A.values(it);
						break;
					}
				}
			}
		}
	}
}
}
}
}
#endif
