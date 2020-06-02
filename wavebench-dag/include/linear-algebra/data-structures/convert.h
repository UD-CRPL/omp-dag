#ifndef __CONVERT_DATA_STRUCTURES_LALG_H__
#define __CONVERT_DATA_STRUCTURES_LALG_H__

#include <set>
#include "../../core/core.h"
#include "../matrix-formats/matrix-formats.h"
#include "lrtree.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __data_structures__ {
template <typename T,typename IT,typename K,typename V,typename Allocator1,typename Allocator2,typename PT>
void convert(MatrixCXS<T,IT,Allocator1,CRS>& adjacency,const LRTree<K,V,Allocator2,PT>& tree) {
	adjacency.resize(tree.size()+1,tree.size()+1);
	adjacency.set();
	std::set<Vector<IT,2>> edges;
	auto addEdge=[&edges,&tree](const Node<K,V,PT>& node,long depth) {
//		if(node.value!=tree.invalidPos)
			edges.insert(Vector<IT,2>({node.key,tree[node.parent].key}));
	};
	tree.traverseLeftRight(addEdge);
	size_t pos=0;
	IT row=0;
	for(auto it=edges.begin();it!=edges.end();++it) {
		auto tmp=*it;
		for(IT i=row;i<tmp[0];++i)
			adjacency.ptr(i+1)=pos;
		row=tmp[0];
		adjacency.indxs(pos)=tmp[1];
		adjacency.values(pos++)=1;
		adjacency.ptr(row+1)=pos;
	}
	adjacency.setNZV(edges.size());
	adjacency.ptr(tree.size()+1)=edges.size();
}
}
}
}
#endif
