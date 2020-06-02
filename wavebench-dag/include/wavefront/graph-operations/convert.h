#ifndef __CONVERT_GRAPH_OPERATIONS_WAVEFRONT_H__
#define __CONVERT_GRAPH_OPERATIONS_WAVEFRONT_H__

#include "../data-structures/data-structures.h"

#include <algorithm>

namespace __core__ {
namespace __wavefront__ {
namespace __graph__ {
template <bool contigous=true,typename AllocatorG=void,typename ATG=void,typename VG=void,typename PTG=int,typename ITG=int,typename VertexType=void,typename EdgeType=void,typename AllocatorCXS=void,typename ITCX=int,enable_IT<contigous==true> =0>
void convert(GraphCXS<VertexType,EdgeType,AllocatorCXS,ITCX>& dstGraph,const Graph<AllocatorG,ATG,VG,PTG,ITG>& srcGraph) {
	dstGraph.resize(srcGraph.vertex_quantity(),srcGraph.edge_quantity());
	dstGraph.set(0);
	size_t pos=0;
	dstGraph.ptr(0)=0;
	for(size_t i=0;i<dstGraph.v();++i) {
		PTG it=srcGraph[i].head;
		while(it!=srcGraph.invalidPos) {
			dstGraph.indxs(pos++)=srcGraph.edge(it).dvertex;
			it=srcGraph.edge(it).next;
		}
		dstGraph.ptr(i+1)=pos;
		if((dstGraph.ptr(i)+1)<dstGraph.ptr(i+1))
			std::sort(dstGraph.indxs()+dstGraph.ptr(i),dstGraph.indxs()+dstGraph.ptr(i+1));
	}
}
template <bool contigous=true,typename AllocatorG=void,typename ATG=void,typename VG=void,typename PTG=int,typename ITG=int,typename VertexType=void,typename EdgeType=void,typename AllocatorCXS=void,typename ITCX=int,enable_IT<contigous==true> =0>
void convert(Graph<AllocatorG,ATG,VG,PTG,ITG>& dstGraph,const GraphCXS<VertexType,EdgeType,AllocatorCXS,ITCX>& srcGraph) {
	dstGraph.clear();
	dstGraph.resize(srcGraph.v(),srcGraph.e(),(StreamType)0);
	size_t pos=0,ipos;
	typedef typename Graph<AllocatorG,ATG,VG,PTG,ITG>::VertexType VT;
	typedef typename Graph<AllocatorG,ATG,VG,PTG,ITG>::EdgeNodeType ET;
	for(size_t i=0;i<srcGraph.v();++i) {
		ipos=pos;
		PTG prev=dstGraph.invalidPos;
		for(ITCX j=srcGraph.ptr(i);j<srcGraph.ptr(i+1);++j) {
			PTG next=(j+1)<srcGraph.ptr(i+1)?(pos+1):dstGraph.invalidPos;
			dstGraph.edge(pos)=ET(i,srcGraph.indxs(j),pos,prev,next);
			prev=pos++;
		}
		if(ipos!=pos)
			dstGraph[i]=VT(i,ipos,pos-1,pos-ipos);
		else
			dstGraph[i]=VT(i,dstGraph.invalidPos,dstGraph.invalidPos,0);
	}
}
}
}
}
#endif
