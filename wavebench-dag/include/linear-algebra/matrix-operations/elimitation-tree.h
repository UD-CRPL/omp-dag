#ifndef __ELIMITATION_TREE_MATRIX_OPERATIONS_H__
#define __ELIMITATION_TREE_MATRIX_OPERATIONS_H__

#include <vector>

#include "../matrix-formats/matrix-formats.h"
#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_operations__ {
template <typename T,typename IT,typename Allocator,SparseCXSFormat frmt> LRTree<IT,IT,Allocator,int> elimitationTree(const MatrixCXS<T,IT,Allocator,frmt>& matrix) {
	LRTree<IT,IT,Allocator,int> tree(matrix.n()+4,-1);
	std::vector<int> vars(matrix.n(),-1);
	Node<IT,IT,int> *node,*nodeRow;
	for(IT row=1;row<matrix.n();++row) {
		for(IT k=matrix.ptr(row);row<matrix.ptr(row+1);++k) {
			IT col=matrix.indxs(k);
			if(col>=row)
				break;
			if(vars[row]==-1) {
				nodeRow=&(tree.insert(row,tree.invalidPos,tree.nilNode));
				vars[row]=nodeRow->self;
			}
			nodeRow=&(tree[vars[row]]);
			if(vars[col]==-1) {
				node=&(tree.insert(col,row,*nodeRow));
				vars[col]=node->self;
			}
			else {
				node=&(tree[vars[col]]);
				IT ancestor=node->value;
				node->value=row;
				bool update=(ancestor!=row);
				while((ancestor!=tree.invalidPos)&&update) {
					node=&(tree[vars[ancestor]]);
					ancestor=node->value;
					node->value=row;
					if(ancestor==row) {
						update=false;
						break;
					}
				}
				if(update)
					tree.moveUnder(node->self,nodeRow->self);
			}
		}
	}
	for(size_t row=0;row<matrix.n();++row)
		if(vars[row]==-1)
			tree.insert(row,tree.invalidPos,tree.nilNode);
	return tree;
}

template <typename IT,typename Allocator> void descendentCount(LRTree<IT,IT,Allocator,int>& tree) {
	typedef Node<IT,IT,int> NT;
	auto set=[](NT& node,int depth){ node.value=0; };
	tree.traverseLeftRight(set);
	NT root=*tree;
	int node=root.left_child;
	bool down=true;
	size_t i=0;
	while((i++)<tree.size()) {
		if(down)
			while(tree[node].left_child!=tree.invalidPos)
				node=tree[node].left_child;
		down=false;
		tree[tree[node].parent].value+=tree[node].value+1;
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
template <typename IT,typename Allocator> std::vector<IT> postOrderTree(LRTree<IT,IT,Allocator,int>& tree) {
	typedef Node<IT,IT,int> NT;
	descendentCount(tree);
	auto fn=[&tree](NT& node,long depth){
		if((node.left_child!=tree.invalidPos)&&(node.left_child!=node.right_child)) {
			IT it=node.left_child;
			do {
				NT ni=tree[it];
				IT ot=ni.right_sibling;
				while(ot!=tree.invalidPos) {
					NT no=tree[ot];
					ot=no.right_sibling;
					if(ni.value<no.value) {
						tree.swapChildren(it,no.self,node.self);
						ni=no;
					}
				}
				it=ni.right_sibling;
			} while(it!=tree.invalidPos);
		}
	};
//	tree.print(std::cerr)<<std::endl;
	tree.traverseLeftRightRoot(fn);
	size_t n=tree.size();
	std::vector<IT> pi(n*2);
	NT root=*tree;
	int node=root.left_child,tmp=0;
	bool down=true;
	size_t i=0;
	while((i++)<n) {
		if(down)
			while(tree[node].left_child!=tree.invalidPos)
				node=tree[node].left_child;
		down=false;
		pi[tmp]=tree[node].key;
		pi[tree[node].key+n]=tmp;
		tree[node].key=tmp++;
		if(tree[node].right_sibling!=tree.invalidPos) {
			node=tree[node].right_sibling;
			down=true;
		}
		else
			node=tree[node].parent;
		if(node==0||node==tree.invalidPos)
			break;
	}
//	tree.print(std::cerr)<<std::endl;
	return pi;
}
/*
 * template <typename IT,typename Allocator> std::vector<IT> postOrderTree(LRTree<IT,IT,Allocator,int>& tree) {
	typedef Node<IT,IT,int> NT;
	size_t n=tree.size();
	std::vector<IT> pi(n*4);
	NT root=*tree;
	int node=root.left_child,tmp=0;
	bool down=true;
	size_t i=0;
	while((i++)<n) {
		if(down)
			while(tree[node].left_child!=tree.invalidPos)
				node=tree[node].left_child;
		down=false;
		pi[tmp]=tree[node].key;
		pi[tree[node].key+n]=tmp;
		if(tree[node].parent!=0)
			pi[(2*n)+tree[tree[node].parent].key]+=pi[(2*n)+tree[node].key]+1;
		tree[node].key=tmp++;
		if(tree[node].right_sibling!=tree.invalidPos) {
			node=tree[node].right_sibling;
			down=true;
		}
		else
			node=tree[node].parent;
		if(node==0||node==tree.invalidPos)
			break;
	}
	for(i=0;i<n;++i)
		pi[3*n+pi[i+n]]=pi[2*n+i];
	return pi;
}
 */
}
}
}
#endif
