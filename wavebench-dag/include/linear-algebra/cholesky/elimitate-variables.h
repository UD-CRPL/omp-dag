#ifndef __ELIMITATE_VARIABLES_H__
#define __ELIMITATE_VARIABLES_H__

#include "../matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <typename T,typename IT>
void eliminateVariable(MatrixCXSHandler<T,IT>& L,MatrixCXSHandler<T,IT>& A,MatrixCXSHandler<IT,IT>& RP,ArrayHandler<IT,IT>& p,ArrayHandler<T,IT>& c,IT k){
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
template <typename T,typename IT>
double eliminateVariableT(MatrixCXSHandler<T,IT>& L,MatrixCXSHandler<T,IT>& A,MatrixCXSHandler<IT,IT>& RP,ArrayHandler<IT,IT>& p,ArrayHandler<T,IT>& c,IT k){
	cpu_timer timer;
	timer.start();
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
	timer.stop();
	return timer.elapsed_time();
}
}
}
}
#endif
