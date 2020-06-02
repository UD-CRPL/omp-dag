#include "dependency-graph.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
Graph<cpuAllocator,void> smithWatermanDG(size_t n,size_t m){
	Graph<cpuAllocator,void> graph(n*m+1,3*((n-1)*(m-1))+n+m-1,-1,DEFAULT_STREAM);
	graph.initVertices();
	graph.initEdges();
	auto coordinates=[&m](size_t i,size_t j){ return i*m+j; };
	for(size_t i=1;i<n;++i)
		graph.insertEdge(graph[coordinates(i-1,0)],graph[coordinates(i,0)]);
	for(size_t i=1;i<m;++i)
		graph.insertEdge(graph[coordinates(0,i-1)],graph[coordinates(0,i)]);
	for(size_t i=1;i<n;++i) {
		for(size_t j=1;j<m;++j) {
			graph.insertEdge(graph[coordinates(i-1,j-1)],graph[coordinates(i,j)]);
			graph.insertEdge(graph[coordinates(i-1,j)],graph[coordinates(i,j)]);
			graph.insertEdge(graph[coordinates(i,j-1)],graph[coordinates(i,j)]);
		}
	}
	graph.insertEdge(graph[coordinates(n-1,m-1)],graph[n*m]);
	return graph;
}
}
}
}
