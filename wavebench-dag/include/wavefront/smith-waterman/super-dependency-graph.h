#ifndef __SUPER_DEPENDENCY_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__
#define __SUPER_DEPENDENCY_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__

#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
struct SmithWatermanVertex {
	int_2A pos;
	int_2A dims;
};
struct SmithWatermanVertexDebug {
	int_2A pos;
	int_2A dims;
	double etime;
	int tid;
};
template <typename T=int> std::ostream& operator<<(std::ostream& ost,const SmithWatermanVertex& v) {
	print(v.pos,ost)<<" ";print(v.dims,ost);
	return ost;
}
template <typename T=int> std::ostream& operator<<(std::ostream& ost,const SmithWatermanVertexDebug& v) {
	print(v.pos,ost)<<" ";print(v.dims,ost)<<" "<<v.etime<<" "<<v.tid;
	return ost;
}
template <typename T=int> GraphCXS<SmithWatermanVertex,float,cpuAllocator> smithWatermanSDG(size_t n,size_t m,size_t px,size_t py){
	size_t nv=__ceil__(static_cast<double>(n)/px), mv=__ceil__(static_cast<double>(m)/py);
	GraphCXS<SmithWatermanVertex,float,cpuAllocator>  graph(nv*mv+1,3*((nv-1)*(mv-1))+nv+mv-1,-1,DEFAULT_STREAM);
	auto coordinates=[](size_t i,size_t j,size_t c) { return i*c+j; };
	auto icoordinates=[](size_t i,size_t c) { return int_2A({i/c,i%c}); };
	size_t pos=0;
	for(size_t i=0;i<nv;++i)
		for(size_t j=0;j<mv;++j) {
			SmithWatermanVertex v;
			v.pos=int_2A({i*px+1,j*py+1});
			v.dims=int_2A({px,py});
			if((j+1)<mv)
				graph.indxs(pos++)=coordinates(i,j+1,mv);
			else
				v.dims[1]=m-j*py;
			if((i+1)<nv)
				graph.indxs(pos++)=coordinates(i+1,j,mv);
			else
				v.dims[0]=n-i*px;
			if(((j+1)<mv)&&((i+1)<nv))
				graph.indxs(pos++)=coordinates(i+1,j+1,mv);
			graph.ptr(coordinates(i,j,mv)+1)=pos;
			graph.vertices(coordinates(i,j,mv))=v;
		}
	graph.ptr(coordinates(nv-1,mv-1,mv)+1)+=1;
	graph.indxs(pos)=coordinates(nv-1,mv-1,mv)+1;
	graph.ptr(coordinates(nv-1,mv-1,mv)+2)=pos+1;
	return graph;
}
template <typename T=int> GraphCXS<SmithWatermanVertexDebug,float,cpuAllocator> smithWatermanSDGD(size_t n,size_t m,size_t px,size_t py){
	size_t nv=__ceil__(static_cast<double>(n)/px), mv=__ceil__(static_cast<double>(m)/py);
	GraphCXS<SmithWatermanVertexDebug,float,cpuAllocator>  graph(nv*mv+1,3*((nv-1)*(mv-1))+nv+mv-1,-1,DEFAULT_STREAM);
	auto coordinates=[](size_t i,size_t j,size_t c) { return i*c+j; };
	auto icoordinates=[](size_t i,size_t c) { return int_2A({i/c,i%c}); };
	size_t pos=0;
	for(size_t i=0;i<nv;++i)
		for(size_t j=0;j<mv;++j) {
			SmithWatermanVertexDebug v;
			v.pos=int_2A({i*px+1,j*py+1});
			v.dims=int_2A({px,py});
			v.etime=0;
			v.tid=-1;
			if((j+1)<mv)
				graph.indxs(pos++)=coordinates(i,j+1,mv);
			else
				v.dims[1]=m-j*py;
			if((i+1)<nv)
				graph.indxs(pos++)=coordinates(i+1,j,mv);
			else
				v.dims[0]=n-i*px;
			if(((j+1)<mv)&&((i+1)<nv))
				graph.indxs(pos++)=coordinates(i+1,j+1,mv);
			graph.ptr(coordinates(i,j,mv)+1)=pos;
			graph.vertices(coordinates(i,j,mv))=v;
		}
	graph.ptr(coordinates(nv-1,mv-1,mv)+1)+=1;
	graph.indxs(pos)=coordinates(nv-1,mv-1,mv)+1;
	graph.ptr(coordinates(nv-1,mv-1,mv)+2)=pos+1;
	return graph;
}
}
}
}
#endif
