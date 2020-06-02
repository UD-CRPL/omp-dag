#ifndef __SPARSE_MAP_CHOLESKY_H__
#define __SPARSE_MAP_CHOLESKY_H__

#include "../matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
template <typename T=void> MatrixMap<T> choleskyInput(const MatrixMap<T>& A) {
	MatrixMap<T> R=A;
	for(size_t k=0;k<R.n();++k) {
		auto kjIt=(*R).find(size_2A({k,k}));
		++kjIt;
		while(kjIt!=(*R).end()) {
			size_2A kj=kjIt->first;
			if(kj[0]!=k)
				break;
			auto kiIt=kjIt;
			++kiIt;
			while(kiIt!=(*R).end()) {
				size_2A ki=kiIt->first;
				if(ki[0]!=k)
					break;
				if((*A).find(size_2A({kj[1],ki[1]}))==(*A).cend())
					R[size_2A({kj[1],ki[1]})]=0;
				kiIt=(*R).upper_bound(ki);
			}
			kjIt=(*R).upper_bound(kj);
		}
	}
	for(auto it=(*R).begin();it!=(*R).end();) {
		if(it->first[0]>it->first[1])
			it=(*R).erase(it);
		else
			++it;
	}
	return R;
}
template <int ver,typename T=void> typename std::enable_if<ver==0,void>::type cholesky(MatrixMap<T>& R) {
	for(size_t k=0;k<R.n();++k) {
		auto RkkI=(*R).find(size_2A({k,k}));
		auto RknknI=(*R).find(size_2A({k+1,k+1}));
		auto tmpI=RkkI;
		++tmpI;
		T Rkk=RkkI->second;
	        for(auto jt=tmpI;jt!=RknknI;++jt)
	            for(auto it=jt;it!=RknknI;++it)
	                R(jt->first[1],it->first[1])-=it->second*(jt->second)/Rkk;
	        T tmp=__sqrt__(__abs__(Rkk));
	        for(auto it=RkkI;it!=RknknI;++it)
	            it->second=it->second/tmp;
	}
}
}
}
}

#endif
