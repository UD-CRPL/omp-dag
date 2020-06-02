#ifndef __TOPOLOGICAL_SORT_GRAPH_OPERATIONS_H__
#define __TOPOLOGICAL_SORT_GRAPH_OPERATIONS_H__

#include <stack>

#include "../../core/core.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __wavefront__ {
namespace __graph__ {
template <typename FT=int,typename VertexType=void,typename EdgeType=void,typename Allocator=void,typename IT=int,typename...Args>
void topologicalSort(GraphCXS<VertexType,EdgeType,Allocator,IT>& graph,FT& fn,std::vector<int>& indegree,Args...args) {
	indegree.assign(graph.v(),0);
	for(size_t i=0;i<graph.e();++i)
		indegree[graph.indxs(i)]+=1;
	std::stack<IT> s;
	for(size_t i=0;i<graph.v();++i)
		if(indegree[i]==0)
			s.push(i);
	while(!s.empty()) {
		IT v=s.top();
		s.pop();
		fn(v,graph,args...);
		for(IT j=graph.ptr(v);j<graph.ptr(v+1);++j) {
			indegree[graph.indxs(j)]-=1;
			if(indegree[graph.indxs(j)]==0)
				s.push(graph.indxs(j));
		}
	}
}
}
}
}
#endif
