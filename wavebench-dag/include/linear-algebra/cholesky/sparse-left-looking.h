#ifndef __SPARSE_LEFT_LOOKING_H__
#define __SPARSE_LEFT_LOOKING_H__

#include "../matrix-formats/matrix-formats.h"
#include "elimitate-variables.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <typename T,typename IT>
void choleskyLeftLooking(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,MatrixCXSHandler<IT,IT> Etree,ArrayHandler<IT,IT> p,ArrayHandler<T,IT> c){
	for(IT k=0;k<A.n();++k) {
		for(IT it=A.ptr(k);it<A.ptr(k+1);++it) {
			IT i=A.indxs(it);
			if(k<=i)
				c[i]=A.values(it);
		}
		for(IT it=RP.ptr(k);it<(RP.ptr(k+1)-1);++it) {
			IT j=RP.indxs(it);
			T lkj=L.values(p[j]);
			for(IT ot=p[j];ot<L.ptr(j+1);++ot) {
				IT i=L.indxs(ot);
				c[i]=c[i]-lkj*L.values(ot);
			}
			p[j]+=1;
		}
		p[k]=L.ptr(k)+1;
		T lkk=__sqrt__(c[k]);
		L.values(L.ptr(k))=lkk;
		for(IT it=L.ptr(k)+1;it<L.ptr(k+1);++it) {
			IT i=L.indxs(it);
			L.values(it)=c[i]/lkk;
			c[i]=0;
		}
	}
}
template <typename T,typename IT,typename Allocator>
void choleskyLeftLooking(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,MatrixCXSHandler<IT,IT> RP,LRTree<IT,IT,Allocator,int>& tree,ArrayHandler<IT,IT> p,ArrayHandler<T,IT> c){
	typedef Node<IT,IT,int> NT;
	NT root=*tree;
	int node=root.left_child;
	bool down=true;
	size_t i=0;
	while((i++)<tree.size()) {
		if(down)
			while(tree[node].left_child!=tree.invalidPos)
				node=tree[node].left_child;
		down=false;
		eliminateVariable(L,A,RP,p,c,tree[node].key);
		if(tree[node].right_sibling!=tree.invalidPos) {
			node=tree[node].right_sibling;
			down=true;
		}
		else
			node=tree[node].parent;
		if(node==0||node==tree.invalidPos)
			break;
	}
}
}
}
}
#endif
