#ifndef __SUPER_NODES_H__
#define __SUPER_NODES_H__

#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
struct SuperNode {
	int parent=-1;
	int evars=0;
	double time=0;
	int processor=0;
	std::vector<int> nodes=std::vector<int>();
	SuperNode(int v) {
	}
	SuperNode(int p,int v,int ev=0) :nodes(std::vector<int>()) {
		parent=p;
		nodes.push_back(v);
		evars=ev;
	}
	void insert(int v) {
		nodes.push_back(v);
	}
};
namespace __super_nodes__ {
template <typename IT,typename Allocator> void simplifyTree(LRTree<IT,IT,Allocator,int>& tree,size_t branchSize,double tolerance=0.2) {
	typedef Node<IT,IT,int> NT;
	IT bbsize=branchSize+tolerance*branchSize;
	bool down=true;
	int node=(*tree).left_child;
	size_t i=0;
	while((i++)<tree.size()) {
		if(down) {
			if(tree[node].value<bbsize) {
				tree.eraseChildren(tree[node],tree[node].value);
				down=false;
			}
			else {
				size_t evars=1;
				while(evars<branchSize) {
					IT it=tree[node].right_child;
					while((it!=tree.invalidPos)&&(evars<branchSize)) {
						NT n=tree[it];
						tree.erase(n);
						it=n.left_sibling;
						++evars;
					}
				}
				down=true;
			}
		}
		if(down&&(tree[node].left_child!=tree.invalidPos))
			node=tree[node].left_child;
		else if(tree[node].right_sibling!=tree.invalidPos) {
			node=tree[node].right_sibling;
			down=true;
		}
		else {
			node=tree[node].parent;
			down=false;
		}
		if(node==0||node==tree.invalidPos)
			break;
	}
}

template <typename IT,typename Allocator> void eraseChildren(LRTree<IT,IT,Allocator,int>& tree,Node<IT,IT,int>& node,std::vector<bool>& removable) {
	removable[node.key]=false;
	int nodeIt=node.left_child;
	bool down=true;
	while((nodeIt!=tree.invalidPos)&&(nodeIt!=node.self)) {
		if(down)
			while(tree[nodeIt].left_child!=tree.invalidPos)
				nodeIt=tree[nodeIt].left_child;
		down=false;
		Node<IT,IT,int> tmp=tree[nodeIt];
		if(removable[tmp.key])
			tree.erase(tree[nodeIt]);
		if(tmp.right_sibling!=tree.invalidPos) {
			nodeIt=tmp.right_sibling;
			down=true;
		}
		else
			nodeIt=tmp.parent;
	}
}
template <typename IT,typename Allocator> void simplifyTreeIteration(LRTree<IT,IT,Allocator,int>& tree,int onode,int maxbranchSize,int minbranchSize,std::vector<int>& rcounts,std::vector<bool>& removable) {
	memset(rcounts.data(),0,rcounts.size()*sizeof(int));
	auto cut=[&tree,&rcounts,&removable](int n,int maxb,int minb) {
		if((tree[n].value<=maxb)&&removable[tree[n].key]) {
//			IT parent=tree[tree[n].parent].key;
			rcounts[tree[n].key]=tree[n].value+1;
//			rcounts[parent]+=tree[n].value+1;
			eraseChildren(tree,tree[n],removable);
			return true;
		}
		return false;
	};
	int node=onode;
	bool down=true,noRm=true;
	do {
		if(down) {
			while(tree[node].left_child!=tree.invalidPos) {
				if(cut(node,maxbranchSize,minbranchSize)) {
					noRm=false;
					break;
				}
				node=tree[node].left_child;
			}
		}
		down=false;
		if(node!=0) {
			tree[tree[node].parent].value-=rcounts[tree[node].key];
			rcounts[tree[tree[node].parent].key]+=rcounts[tree[node].key];
		}
		if(tree[node].right_sibling!=tree.invalidPos) {
			node=tree[node].right_sibling;
			down=true;
		}
		else
			node=tree[node].parent;
	} while((node!=tree.invalidPos)&&(node!=onode)&&(node!=0));
	if(noRm)
		removable[tree[onode].key]=false;
}
template <int v,typename IT,typename Allocator> void simplifyTree(LRTree<IT,IT,Allocator,int>& tree,size_t branchSize,double tolerance=0.2) {
	size_t maxbs=__ceil__((1.+tolerance)*branchSize),minbs=__floor__((1.-tolerance)*branchSize);
	std::vector<int> rcounts(tree.size()+2,0);
	std::vector<bool> removable(tree.size()+2,true);
	int node=(*tree).left_child;
	bool all=true;
	size_t maxit=8*tree.size()/branchSize;
	do {
		all=true;
		node=(*tree).left_child;
		while(node!=tree.invalidPos) {
			if(removable[tree[node].key])
				simplifyTreeIteration(tree,node,maxbs,minbs,rcounts,removable);
			all=all&&!removable[tree[node].key];
			node=tree[node].right_sibling;
		}
		maxbs=(maxbs+minbs)/2;
		maxit--;
	} while((!all)&&(maxit!=0));
//	std::cerr<<"\t\tMax it: "<<maxit<<std::endl;
}
template <typename IT,typename Allocator> void populateTree(LRTree<IT,SuperNode,Allocator,int>& sntree,LRTree<IT,IT,Allocator,int>& simplifiedTree) {
	size_t pos=1;
	int node=(*simplifiedTree).left_child;
	bool down=true;
	size_t i=0;
	Node<IT,SuperNode,int>* n=&(*sntree);
	while((i++)<simplifiedTree.size()) {
		if(down) {
			n=&sntree.insert(pos++,SuperNode(n->key,simplifiedTree[node].self,simplifiedTree[node].value+1),*n);
			while(simplifiedTree[node].left_child!=simplifiedTree.invalidPos) {
				node=simplifiedTree[node].left_child;
				n=&sntree.insert(pos++,SuperNode(n->key,simplifiedTree[node].self,simplifiedTree[node].value+1),*n);
			}
		}
		down=false;
		if(simplifiedTree[node].right_sibling!=simplifiedTree.invalidPos) {
			node=simplifiedTree[node].right_sibling;
			n=&(sntree[n->parent]);
			down=true;
		}
		else {
			node=simplifiedTree[node].parent;
			n=&(sntree[n->parent]);
		}
		if(node==0||node==simplifiedTree.invalidPos)
			break;
	}
}
}
std::ostream& operator<<(std::ostream& ost,const SuperNode& node);
template <typename IT,typename Allocator> auto superNodes(LRTree<IT,IT,Allocator,int>& tree,size_t branchSize,double tolerance=0.2) {
	LRTree<IT,IT,Allocator,int> simplifiedTree(tree);
//	tree.print(std::cerr)<<std::endl;
	__super_nodes__::simplifyTree<0>(simplifiedTree,branchSize,tolerance);
//	simplifiedTree.print(std::cerr)<<std::endl;
//	std::cerr<<std::endl<<simplifiedTree.size()<<std::endl;
	LRTree<IT,SuperNode,Allocator,int> sntree(simplifiedTree.size(),-1);
	__super_nodes__::populateTree(sntree,simplifiedTree);
//	int node=(*sntree).left_child;
//	bool down=true;
//	size_t i=0;
//	while((i++)<sntree.size()) {
//		if(down)
//			while(sntree[node].left_child!=sntree.invalidPos)
//				node=sntree[node].left_child;
//		down=false;
//		//(sntree[node].left_child==sntree.invalidPos)&&
//		if((sntree[node].left_sibling==sntree.invalidPos)&&(sntree[node].right_sibling==sntree.invalidPos)&&(sntree[node].left_child==sntree.invalidPos)&&(sntree[node].parent!=0)) {
//			Node<IT,SuperNode,int>& tmp=sntree[node];
//			node=tmp.parent;
//			sntree.erase(tmp);
//		}
//		else {
//			if(sntree[node].right_sibling!=sntree.invalidPos) {
//				size_t evars=sntree[node].value.evars;
//				int it=sntree[node].right_sibling;
//				while(it!=sntree.invalidPos) {
//					if((evars+sntree[it].value.evars)<=branchSize) {
//						sntree[node].value.evars+=sntree[it].value.evars;
//						sntree[node].value.nodes.insert(sntree[node].value.nodes.end(),sntree[it].value.nodes.begin(),sntree[it].value.nodes.end());
//						evars+=sntree[it].value.evars;
//						int tmp=sntree[it].right_sibling;
//						sntree.erase(sntree[it]);
//						it=tmp;
//						continue;
//					}
//					it=sntree[it].right_sibling;
//				}
//				if(sntree[node].right_sibling!=sntree.invalidPos) {
//					node=sntree[node].right_sibling;
//					down=true;
//				}
//			}
//			else
//				node=sntree[node].parent;
//		}
//		if(node==0||node==sntree.invalidPos)
//			break;
//	}
//	auto rc=[&sntree](Node<int,SuperNode,int> &n,long depth){
//		int tmp=n.left_child;
//		while(tmp!=sntree.invalidPos) {
//			n.value.evars-=sntree[tmp].value.evars;
//			tmp=sntree[tmp].right_sibling;
//		}
//	};
//	sntree.traverseLeftRight(rc);
	return sntree;
}
template <typename IT,typename Allocator> auto superNodes(LRTree<IT,IT,Allocator,int>& tree,LRTree<IT,IT,Allocator,int>& simplifiedTree,size_t branchSize,double tolerance=0.2) {
	simplifiedTree.import(tree);
	__super_nodes__::simplifyTree(simplifiedTree,branchSize,tolerance);
	LRTree<IT,SuperNode,Allocator,int> sntree(simplifiedTree.size(),-1);
	__super_nodes__::populateTree(sntree,simplifiedTree);
	int node=(*sntree).left_child;
	bool down=true;
	size_t i=0;
	while((i++)<sntree.size()) {
		if(down)
			while(sntree[node].left_child!=sntree.invalidPos)
				node=sntree[node].left_child;
		down=false;
		if((sntree[node].left_sibling==sntree.invalidPos)&&(sntree[node].right_sibling==sntree.invalidPos)&&(sntree[node].left_child==sntree.invalidPos)&&(sntree[node].parent!=0)) {
			Node<IT,SuperNode,int>& tmp=sntree[node];
			node=tmp.parent;
			sntree.erase(tmp);
		}
		else {
			if(sntree[node].right_sibling!=sntree.invalidPos) {
				size_t evars=sntree[node].value.evars;
				int it=sntree[node].right_sibling;
				while(it!=sntree.invalidPos) {
					if((evars+sntree[it].value.evars)<=branchSize) {
						sntree[node].value.evars+=sntree[it].value.evars;
						sntree[node].value.nodes.insert(sntree[node].value.nodes.end(),sntree[it].value.nodes.begin(),sntree[it].value.nodes.end());
						evars+=sntree[it].value.evars;
						int tmp=sntree[it].right_sibling;
						sntree.erase(sntree[it]);
						it=tmp;
						continue;
					}
					it=sntree[it].right_sibling;
				}
				if(sntree[node].right_sibling!=sntree.invalidPos) {
					node=sntree[node].right_sibling;
					down=true;
				}
			}
			else
				node=sntree[node].parent;
		}
		if(node==0||node==sntree.invalidPos)
			break;
	}
	return sntree;
}
}
}
}
#endif
