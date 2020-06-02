#ifndef __DEPTH_FIRST_GRAPH_OPERATIONS_H__
#define __DEPTH_FIRST_GRAPH_OPERATIONS_H__

#include <stack>

#include "../../core/core.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __wavefront__ {
namespace __graph__ {
template <typename FT=int,typename Allocator=void,typename AT=void,typename V=void,typename PT=int,typename IT=int,typename...Args>
void depthFirst(Graph<Allocator,AT,V,PT,IT>& graph,FT& fn,Vertex<AT,PT,IT> sourceVertex,std::vector<bool>& discovered,Args...args) {
	if((sourceVertex.self<graph.vertices().capacity())&&(sourceVertex.self>=0)) {
		std::stack<PT> s;
		discovered.assign(graph.vertices().capacity(),false);
		s.push(sourceVertex.self);
		while(!s.empty()) {
			Vertex<AT,PT,IT> v=graph[s.top()];
			s.pop();
			if(v.self!=graph.invalidPos)
				if(!discovered[v.self]) {
					fn(v,args...);
					discovered[v.self]=true;
					PT it=v.head;
					while(it!=graph.invalidPos) {
						s.push(graph.edge(it).dvertex);
						it=graph.edge(it).next;
					}
				}
		}
	}
}
}
}
}
#endif
