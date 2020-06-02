#ifndef __CONVERT_MATRIX_FORMATS_H__
#define __CONVERT_MATRIX_FORMATS_H__

#include "dense-matrix.h"
#include "sparse-cxs.h"
#include "sparse-map.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
template <typename T,typename AllocatorT1,typename AllocatorT2,uint alignment,typename IT,SparseCXSFormat frmt,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void convert(MatrixCXS<T,IT,AllocatorT2,frmt>& dmatrix,const Matrix<T,AllocatorT1,alignment>& smatrix,T eps=1e-12) {
	size_t nzv=0;
	for(size_t i=0;i<smatrix.n();++i)
		for(size_t j=0;j<smatrix.m();++j)
			if(__abs__(smatrix(i,j))>eps)
				++nzv;
	dmatrix.resize(smatrix.rows(),smatrix.cols(),nzv);
	dmatrix.set(0);
	IT ptrPos=0;
	switch(frmt) {
	case COO:
		for(size_t i=0;i<smatrix.n();++i) {
			for(size_t j=0;j<smatrix.m();++j) {
				if(__abs__(smatrix(i,j))>eps){
					dmatrix.ptr(ptrPos)=i;
					dmatrix.indxs(ptrPos)=j;
					dmatrix.values(ptrPos++)=smatrix(i,j);
				}
			}
		}
		break;
	case CRS:
		for(size_t i=0;i<smatrix.n();++i) {
			for(size_t j=0;j<smatrix.m();++j) {
				if(__abs__(smatrix(i,j))>eps){
					dmatrix.indxs(ptrPos)=j;
					dmatrix.values(ptrPos++)=smatrix(i,j);
				}
			}
			dmatrix.ptr(i+1)=ptrPos;
		}
		break;
	case CCS:
		for(size_t j=0;j<smatrix.m();++j) {
			for(size_t i=0;i<smatrix.n();++i) {
				if(__abs__(smatrix(i,j))>eps){
					dmatrix.indxs(ptrPos)=i;
					dmatrix.values(ptrPos++)=smatrix(i,j);
				}
			}
			dmatrix.ptr(j+1)=ptrPos;
		}
		break;
	}
}
template <typename T,typename AllocatorT1,typename AllocatorT2,uint alignment,typename IT,SparseCXSFormat frmt,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void convert(Matrix<T,AllocatorT1,alignment>& dmatrix,const MatrixCXS<T,IT,AllocatorT2,frmt>& smatrix) {
	dmatrix.resize(smatrix.rows(),smatrix.cols(),0);
	dmatrix=0;
	switch(frmt) {
		case COO:
			for(size_t i=0;i<smatrix.nzv();++i)
				dmatrix(smatrix.ptr(i),smatrix.indxs(i))=smatrix.values(i);
			break;
		case CRS:
			for(size_t i=0;i<smatrix.n();++i)
				for(size_t j=smatrix.ptr(i);j<smatrix.ptr(i+1);++j)
					dmatrix(i,smatrix.indxs(j))=smatrix.values(j);
			break;
		case CCS:
			for(size_t j=0;j<smatrix.m();++j)
				for(size_t i=smatrix.ptr(j);i<smatrix.ptr(j+1);++i)
					dmatrix(smatrix.indxs(i),j)=smatrix.values(i);
			break;
		}
}
template <typename T,typename AllocatorT,uint alignment,enable_IT<eq_CE(AllocatorT::location,HOST)> = 0>
void convert(MatrixMap<T>& dmatrix,const Matrix<T,AllocatorT,alignment>& smatrix,T eps=1e-12) {
	dmatrix.clear();
	dmatrix.reshape(smatrix.rows(),smatrix.cols());
	for(size_t i=0;i<smatrix.n();++i)
		for(size_t j=0;j<smatrix.m();++j)
			if(__abs__(smatrix(i,j))>eps)
				dmatrix(i,j)=smatrix(i,j);
}
template <typename T,typename AllocatorT,uint alignment,enable_IT<eq_CE(AllocatorT::location,HOST)> = 0>
void convert(Matrix<T,AllocatorT,alignment>& dmatrix,const MatrixMap<T>& smatrix) {
	dmatrix.resize(smatrix.rows(),smatrix.cols(),0);
	dmatrix=0;
	for(auto it=(*smatrix).cbegin();it!=(*smatrix).cend();++it)
		dmatrix(it->first[0],it->first[1])=it->second;
}
template <typename T,typename AllocatorT,typename IT,SparseCXSFormat frmt,enable_IT<eq_CE(AllocatorT::location,HOST)> = 0>
void convert(MatrixCXS<T,IT,AllocatorT,frmt>& dmatrix,const MatrixMap<T>& smatrix) {
	dmatrix.resize(smatrix.rows(),smatrix.cols(),smatrix.nzv());
	dmatrix.set(0);
	IT ptrPos=0;
	switch(frmt) {
	case COO:
		for(auto it=(*smatrix).cbegin();it!=(*smatrix).cend();++it) {
			dmatrix.ptr(ptrPos)=it->first[0];
			dmatrix.indxs(ptrPos)=it->first[1];
			dmatrix.values(ptrPos++)=it->second;
		}
		break;
	case CRS:
		IT pastRow=0;
		for(auto it=(*smatrix).cbegin();it!=(*smatrix).cend();++it) {
			dmatrix.indxs(ptrPos)=it->first[1];
			dmatrix.values(ptrPos++)=it->second;
			dmatrix.ptr(it->first[0]+1)=ptrPos;
			for(auto j=pastRow+1;j<=it->first[0];++j)
				dmatrix.ptr(j+1)=ptrPos;
			pastRow=it->first[0];
		}
		break;
	}
}
template <typename T,typename AllocatorT,typename IT,SparseCXSFormat frmt,enable_IT<eq_CE(AllocatorT::location,HOST)> = 0>
void convert(MatrixMap<T>& dmatrix,const MatrixCXS<T,IT,AllocatorT,frmt>& smatrix) {
	dmatrix.clear();
	dmatrix.reshape(smatrix.rows(),smatrix.cols());
	switch(frmt) {
	case COO:
		for(size_t i=0;i<smatrix.nzv();++i)
			dmatrix(smatrix.ptr(i),smatrix.indxs(i))=smatrix.values(i);
		break;
	case CRS:
		for(size_t i=0;i<smatrix.n();++i)
			for(size_t j=smatrix.ptr(i);j<smatrix.ptr(i+1);++j)
				dmatrix(i,smatrix.indxs(j))=smatrix.values(j);
		break;
	case CCS:
		for(size_t j=0;j<smatrix.m();++j)
			for(size_t i=smatrix.ptr(j);i<smatrix.ptr(j+1);++i)
				dmatrix(smatrix.indxs(i),j)=smatrix.values(i);
		break;
	}
}
template <typename T,typename AllocatorT1,typename AllocatorT2,typename IT,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void convert(MatrixCXS<T,IT,AllocatorT2,CRS>& dmatrix,const MatrixCXS<T,IT,AllocatorT1,CCS>& smatrix) {
	dmatrix.resize(smatrix.n(),smatrix.m(),smatrix.nzv());
	std::vector<IT> rowCount(dmatrix.rows());
	for(size_t i=0;i<dmatrix.nzv();++i)
		rowCount[smatrix.indxs(i)]+=1;
	dmatrix.ptr(0)=0;
	for(size_t i=0;i<dmatrix.n();++i)
		dmatrix.ptr(i+1)=rowCount[i]+dmatrix.ptr(i);
	for(size_t j=0;j<smatrix.m();++j) {
		for(IT i=smatrix.ptr(j);i<smatrix.ptr(j+1);++i) {
			IT row=smatrix.indxs(i);
			IT ptrPos=dmatrix.ptr(row+1)-rowCount[row];
			rowCount[row]-=1;
			dmatrix.indxs(ptrPos)=j;
			dmatrix.values(ptrPos)=smatrix.values(i);
		}
	}
}
template <typename T,typename AllocatorT1,typename AllocatorT2,typename IT,enable_IT<eq_CE(AllocatorT1::location,AllocatorT2::location)&&eq_CE(AllocatorT1::location,HOST)> = 0>
void convert(MatrixCXS<T,IT,AllocatorT2,CCS>& dmatrix,const MatrixCXS<T,IT,AllocatorT1,CRS>& smatrix) {
	dmatrix.resize(smatrix.n(),smatrix.m(),smatrix.nzv());
	std::vector<IT> colCount(dmatrix.cols());
	for(size_t i=0;i<dmatrix.nzv();++i)
		colCount[smatrix.indxs(i)]+=1;
	dmatrix.ptr(0)=0;
	for(size_t i=0;i<dmatrix.m();++i)
		dmatrix.ptr(i+1)=colCount[i]+dmatrix.ptr(i);
	for(size_t i=0;i<smatrix.n();++i) {
		for(IT j=smatrix.ptr(i);j<smatrix.ptr(i+1);++j) {
			IT col=smatrix.indxs(j);
			IT ptrPos=dmatrix.ptr(col+1)-colCount[col];
			colCount[col]-=1;
			dmatrix.indxs(ptrPos)=i;
			dmatrix.values(ptrPos)=smatrix.values(j);
		}
	}
}
}
}
}
#endif
