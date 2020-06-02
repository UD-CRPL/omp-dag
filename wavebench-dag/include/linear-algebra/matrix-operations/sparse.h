#ifndef __SPARSE_MATRIX_OPERATIONS_H__
#define __SPARSE_MATRIX_OPERATIONS_H__

#include "../matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T>
void transpose(MatrixMap<T>& dmatrix,const MatrixMap<T>& smatrix) {
	dmatrix.clear();
	dmatrix.reshape(smatrix.cols(),smatrix.rows());
	for(auto it=(*smatrix).cbegin();it!=(*smatrix).cend();++it)
		dmatrix(it->first[1],it->first[0])=it->second;
}
template <typename T>
MatrixMap<T> transpose(const MatrixMap<T>& matrix) {
	MatrixMap<T> result;
	transpose(result,matrix);
	return result;
}
template <typename T>
void multiply(MatrixMap<T>& matrix,const MatrixMap<T>& A,const MatrixMap<T>& B) {
	error(A.cols()==B.rows(),"Invalid shapes for the matrices.",RUNTIME_ERROR,throw_error);
	matrix.clear();
	matrix.reshape(A.rows(),B.cols());
	for(auto it=(*A).cbegin();it!=(*A).cend();++it) {
		for(auto jt=(*B).lower_bound(size_2A({it->first[1],0}));jt!=(*B).cend();++jt) {
			if(jt->first[0]==it->first[1])
				matrix(it->first[0],jt->first[1])+=it->second*jt->second;
			else
				break;
		}
	}
}
template <typename T>
MatrixMap<T> multiply(const MatrixMap<T>& A,const MatrixMap<T>& B) {
	MatrixMap<T> result;
	multiply(result,A,B);
	return result;
}
template <typename T>
T maxDifference(const MatrixMap<T>& A,const MatrixMap<T>& B) {
	error(A.cols()==B.cols()&&A.rows()==B.rows(),"Invalid shapes for the matrices.",RUNTIME_ERROR,throw_error);
	MatrixMap<T> tmp=A;
	T result=0;
	for(auto it=(*B).cbegin();it!=(*B).cend();++it)
		tmp[it->first]-=it->second;
	for(auto it=(*tmp).cbegin();it!=(*tmp).cend();++it)
		result=__max__(result,__abs__(it->second));
	return result;
}

template <typename T,typename IT,typename AllocatorD,typename AllocatorS,typename PIT>
void reorderMatrix(MatrixCXS<T,IT,AllocatorD,CCS>& AR,const MatrixCXS<T,IT,AllocatorS,CCS>& A,PIT* permutation) {
	std::vector<IT> invPermutation(A.n(),0);
	for(IT i=0;i<invPermutation.size();++i)
		invPermutation[permutation[i]]=i;
	AR.resize(A.n(),A.m(),A.nzv(),DEFAULT_STREAM);
	AR.set(0);
	for(size_t i=0;i<AR.m();++i)
		AR.ptr(i+1)=(A.ptr(permutation[i]+1)-A.ptr(permutation[i]))+AR.ptr(i);
	for(size_t i=0;i<AR.m();++i) {
		int tmp=0;
		IT pcol=permutation[i];
		std::map<IT,int> cols;
		for(IT it=A.ptr(pcol);it<A.ptr(pcol+1);++it)
			cols.insert(std::make_pair(invPermutation[A.indxs(it)],it));
		for(auto it=cols.begin();it!=cols.end();++it) {
			AR.indxs(tmp+AR.ptr(i))=it->first;
			AR.values(tmp+AR.ptr(i))=A.values(it->second);
			++tmp;
		}
	}
}
template <typename T,typename IT,typename AllocatorD,typename AllocatorS,typename PIT>
void reorderMatrix(MatrixCXS<T,IT,AllocatorD,CCS>& AR,const MatrixCXS<T,IT,AllocatorS,CCS>& A,PIT* permutation,PIT* invPermutation) {
	AR.resize(A.n(),A.m(),A.nzv(),DEFAULT_STREAM);
	AR.set(0);
	for(size_t i=0;i<AR.m();++i)
		AR.ptr(i+1)=(A.ptr(permutation[i]+1)-A.ptr(permutation[i]))+AR.ptr(i);
	for(size_t i=0;i<AR.m();++i) {
		int tmp=0;
		IT pcol=permutation[i];
		std::map<IT,int> cols;
		for(IT it=A.ptr(pcol);it<A.ptr(pcol+1);++it)
			cols.insert(std::make_pair(invPermutation[A.indxs(it)],it));
		for(auto it=cols.begin();it!=cols.end();++it) {
			AR.indxs(tmp+AR.ptr(i))=it->first;
			AR.values(tmp+AR.ptr(i))=A.values(it->second);
			++tmp;
		}
	}
}
template <typename T,typename IT,typename Allocator>
void trimDiagonal(MatrixCXS<T,IT,Allocator,CCS>& R,const MatrixCXS<T,IT,Allocator,CCS>& A) {
	R.resize(A.n(),A.m(),A.nzv());
	R.set(0);
	size_t pos=0;
	for(IT i=0;i<A.m();++i) {
		for(IT it=A.ptr(i);it<A.ptr(i+1);++it) {
			IT r=A.indxs(it);
			if(r!=i) {
				R.indxs(pos)=r;
				R.values(pos)=A.values(it);
				++pos;
			}
		}
		R.ptr(i+1)=pos;
	}
	R.setNZV(R.ptr(A.n()));
}
//		for(IT it=A.ptr(pcol);it<A.ptr(pcol+1);++it) {
//			AR.indxs(AR.ptr(i)+tmp)=invPermutation[A.indxs(it)];
//			AR.values(AR.ptr(i)+tmp)=A.values(it);
//			++tmp;
//		}
}
}
}
#endif
