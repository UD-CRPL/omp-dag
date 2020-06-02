#ifndef __LRTREE_DATA_STRUCTURES_LALG_H__
#define __LRTREE_DATA_STRUCTURES_LALG_H__

#include "../../core/core.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __data_structures__ {
template <typename K,typename V,typename PT=int> struct Node {
	PT self;
	PT parent;
	PT left_child;
	PT right_child;
	PT left_sibling;
	PT right_sibling;
	K key;
	V value;
	Node() {
	}
	constexpr Node(PT s,PT p,PT lc,PT rc,PT ls,PT rs): self(s),parent(p),left_child(lc),right_child(rc),
			left_sibling(ls),right_sibling(rs) {
	}
	constexpr Node(PT s,PT p,PT lc,PT rc,PT ls,PT rs,K k,V v=V(0)): self(s),parent(p),left_child(lc),right_child(rc),
			left_sibling(ls),right_sibling(rs),key(k),value(v) {
	}
	__host_device__ bool operator==(const Node& node) {
		return (parent==node.parent)&&(left_child==node.left_child)&&(right_child==node.right_child)&&(left_sibling==node.left_sibling)&&(right_sibling==node.right_sibling)
				&&(key==node.key);
	}
};
template <typename K,typename PT> struct Node<K,void,PT> {
	PT self;
	PT parent;
	PT left_child;
	PT right_child;
	PT left_sibling;
	PT right_sibling;
	K key;
	Node() {
	}
	constexpr Node(PT s,PT p,PT lc,PT rc,PT ls,PT rs): self(s),parent(p),left_child(lc),right_child(rc),left_sibling(ls),right_sibling(rs) {
	}
	constexpr Node(PT s,PT p,PT lc,PT rc,PT ls,PT rs,K k): self(s),parent(p),left_child(lc),right_child(rc),left_sibling(ls),right_sibling(rs),key(k) {
	}
	__host_device__ bool operator==(const Node& node) {
		return (parent==node.parent)&&(left_child==node.left_child)&&(right_child==node.right_child)&&(left_sibling==node.left_sibling)&&(right_sibling==node.right_sibling)
				&&(key==node.key);
	}
};
template <typename K,typename V,typename Allocator,typename PT=int> class  LRTree {
public:
	typedef K keyType;
	typedef V valueType;
	typedef Node<K,V,PT> nodeType;
	typedef Allocator allocator_type;
	typedef typename Allocator::memory_type memory_type;
	static constexpr PT invalidPos=is_signed_CE<PT>()?-1:__numeric_limits__<PT>::max;
	static const Node<K,V,PT> nilNode;
	static bool isNil(const Node<K,V,PT>& node);
private:
	Array<nodeType,Allocator> __data__;
	size_t __size__=0;
	size_t __pos__=0;
	template <typename KT,typename VT,typename AT,typename PTT> friend class LRTree;
	void increaseSize();
public:
	LRTree(int dev=-1);
	LRTree(LRTree&& tree);
	LRTree(const LRTree& tree);
	LRTree(size_t n,int dev,StreamType stream=0,const Allocator& allocator=Allocator());
	template <typename allocatorType_T> LRTree(const LRTree<K,V,allocatorType_T>& tree,int dev=-1,StreamType stream=0,const Allocator& allocator=Allocator());
	~LRTree();

	LRTree& operator=(LRTree&& tree);
	LRTree& operator=(const LRTree& tree);
	template <typename allocatorType_T> LRTree& operator=(const LRTree<K,V,allocatorType_T>& tree);

	Node<K,V,PT>& operator*();
	const Node<K,V,PT>& operator*() const;
	Node<K,V,PT>& operator[](size_t i);
	const Node<K,V,PT>& operator[](size_t i) const;
	size_t size() const;

	void free();
	void clear();
	void resize(size_t n,StreamType stream=0);

	Node<K,V,PT>& find(K key,const Node<K,V,PT>& node=nilNode);
	const Node<K,V,PT>& find(K key) const;
	template <typename VT=V,enable_IT<is_same_CE<VT,void>()> = 0> Node<K,V,PT>& insert(K key,const Node<K,V,PT>& parent);
	template <typename VT=V,enable_IT<!is_same_CE<VT,void>()> = 0> Node<K,V,PT>& insert(K key,VT val,const Node<K,V,PT>& parent);
	void erase(const Node<K,V,PT>& node);
	void eraseBranch(const Node<K,V,PT>& node);
	void eraseBranch(const Node<K,V,PT>& node,size_t rsize);
	void eraseChildren(const Node<K,V,PT>& node,size_t rsize=0);
	void moveUnder(PT node,PT parent);
	void swapChildren(PT nodex,PT nodey,PT parent);

	template <bool check=true,typename allocatorType_T=void> LRTree& import(const LRTree<K,V,allocatorType_T,PT>& tree,StreamType stream=0);

	template <typename FT,typename...Args> void traverseLeftRight(FT& function,const Node<K,V,PT>& node=nilNode,Args... args);
	template <typename FT,typename...Args> void traverseLeftRight(FT& function,const Node<K,V,PT>& node=nilNode,Args... args) const;

	template <typename FT,typename...Args> void traverseLeftRightRoot(FT& function,const Node<K,V,PT>& node=nilNode,Args... args);
	template <typename FT,typename...Args> void traverseLeftRightRoot(FT& function,const Node<K,V,PT>& node=nilNode,Args... args) const;

	std::ostream& print(std::ostream& ost,const Node<K,V,PT>& node=nilNode,char bbegin='\t') const;
};
template<typename K,typename V,typename Allocator,typename PT> const Node<K,V,PT> LRTree<K,V,Allocator,PT>::nilNode=Node<K,V,PT>(invalidPos,invalidPos,invalidPos,invalidPos,invalidPos,invalidPos,K(0));
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>::LRTree(int dev): __data__(dev) {
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>::LRTree(LRTree&& tree): __data__(std::move(tree.__data__)) {
	move(__size__,tree.__size__);
	move(__pos__,tree.__pos__);
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>::LRTree(const LRTree& tree): __data__(tree.__data__), __size__(tree.__size__), __pos__(tree.__pos__) {
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>::LRTree(size_t n,int dev,StreamType stream,const Allocator& allocator): __data__(Array<nodeType,Allocator>(n+1,dev,0,stream,allocator)) {
	if((memory_type::location==HOST)||(memory_type::location==MANAGED)) {
		__data__[0]=nilNode;
		__data__[0].self=0;
		__pos__=1;
	}
}
template<typename K,typename V,typename Allocator,typename PT> template<typename allocatorType_T>
LRTree<K,V,Allocator,PT>::LRTree(const LRTree<K,V,allocatorType_T>& tree,int dev,StreamType stream,const Allocator& allocator):
	__data__(Array<nodeType,Allocator>(tree.__data__,dev,stream,allocator)), __size__(tree.__size__), __pos__(tree.__pos__) {
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>::~LRTree() {
	__size__=0;
	__pos__=0;
	__data__.free();
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>& LRTree<K,V,Allocator,PT>::operator =(LRTree&& tree) {
	__data__=std::move(tree.__data__);
	move(__size__,tree.__size__);
	move(__pos__,tree.__pos__);
	return *this;
}
template<typename K,typename V,typename Allocator,typename PT>
LRTree<K,V,Allocator,PT>& LRTree<K,V,Allocator,PT>::operator =(const LRTree& tree) {
	return import(tree);
}
template<typename K,typename V,typename Allocator,typename PT> template<typename allocatorType_T>
LRTree<K,V,Allocator,PT>& LRTree<K,V,Allocator,PT>::operator =(const LRTree<K,V,allocatorType_T>& tree) {
	return import(tree);
}
template<typename K,typename V,typename Allocator,typename PT>
Node<K,V,PT>& LRTree<K,V,Allocator,PT>::operator *() {
	return __data__[0];
}
template<typename K,typename V,typename Allocator,typename PT>
const Node<K,V,PT>& LRTree<K,V,Allocator,PT>::operator *() const {
	return __data__[0];
}
template<typename K,typename V,typename Allocator,typename PT>
Node<K,V,PT>& LRTree<K,V,Allocator,PT>::operator[](size_t i) {
	return __data__[i];
}
template<typename K,typename V,typename Allocator,typename PT>
const Node<K,V,PT>& LRTree<K,V,Allocator,PT>::operator[](size_t i) const {
	return __data__[i];
}
template<typename K,typename V,typename Allocator,typename PT>
bool LRTree<K,V,Allocator,PT>::isNil(const Node<K,V,PT>& node) {
	return ((invalidPos==node.parent)&&(invalidPos==node.left_child)&&(invalidPos==node.right_child)&&(invalidPos==node.left_sibling)&&(invalidPos==node.right_sibling))||(invalidPos==node.self);
}
template<typename K,typename V,typename Allocator,typename PT>
size_t LRTree<K,V,Allocator,PT>::size() const {
	return __size__;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::free() {
	__size__=0;
	__pos__=0;
	__data__.free();
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::clear() {
	__size__=0;
	if(((memory_type::location==HOST)||(memory_type::location==MANAGED))&&(__data__.capacity()>0)) {
		__data__[0]=nilNode;
		__data__[0].self=0;
		__pos__=1;
	}
	else
		__pos__=0;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::resize(size_t n,StreamType stream) {
	__data__.reserve(n,stream);
	__data__.expand();
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::increaseSize() {
	if(__data__.capacity()==0) {
		resize(1024);
		if((memory_type::location==HOST)||(memory_type::location==MANAGED)) {
			__data__[0]=nilNode;
			__data__[0].self=0;
			__pos__=1;
		}
	}
	else if(__data__.capacity()==(__pos__+1)) {
		__data__.reserve(((__data__.capacity()+2)*3)/2);
		__data__.expand();
	}
}
template<typename K,typename V,typename Allocator,typename PT>
Node<K,V,PT>& LRTree<K,V,Allocator,PT>::find(K key,const Node<K,V,PT>& node) {
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[0];
	else
		if(node.key==key)
			return __data__[node.self];
	PT it=n.left_child;
	while(it!=invalidPos) {
		if(__data__[it].key==key)
			return __data__[it];
		if(__data__[it].left_child!=invalidPos)
			it=__data__[it].left_child;
		else if(__data__[it].right_sibling!=invalidPos)
			it=__data__[it].right_sibling;
		else {
			PT ot=__data__[it].parent;
			while(ot!=invalidPos) {
				PT tmp=__data__[ot].right_sibling;
				if(tmp!=invalidPos) {
					ot=tmp;
					break;
				}
				else
					ot=__data__[ot].parent;
			}
			it=ot;
		}
	}
	return __data__[0];
}
template<typename K,typename V,typename Allocator,typename PT> template<typename VT,enable_IT<is_same_CE<VT,void>()>>
Node<K,V,PT>& LRTree<K,V,Allocator,PT>::insert(K key,const Node<K,V,PT>& parent) {
	Node<K,V,PT> p=parent;
	increaseSize();
	if(isNil(p))
		p=__data__[0];
	__data__[__pos__]=Node<K,V,PT>(__pos__,p.self,invalidPos,invalidPos,p.right_child,invalidPos,key);
	if(p.right_child!=invalidPos)
		__data__[p.right_child].right_sibling=__pos__;
	__data__[p.self].right_child=__pos__;
	if(p.left_child==invalidPos)
		__data__[p.self].left_child=__pos__;
	++__size__;
	return __data__[__pos__++];
}
template<typename K,typename V,typename Allocator,typename PT> template<typename VT,enable_IT<!is_same_CE<VT,void>()>>
Node<K,V,PT>& LRTree<K,V,Allocator,PT>::insert(K key,VT val,const Node<K,V,PT>& parent) {
	Node<K,V,PT> p=parent;
	increaseSize();
	if(isNil(p))
		p=__data__[0];
	__data__[__pos__]=Node<K,V,PT>(__pos__,p.self,invalidPos,invalidPos,p.right_child,invalidPos,key,val);
	if(p.right_child!=invalidPos)
		__data__[p.right_child].right_sibling=__pos__;
	__data__[p.self].right_child=__pos__;
	if(p.left_child==invalidPos)
		__data__[p.self].left_child=__pos__;
	++__size__;
	return __data__[__pos__++];
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::erase(const Node<K,V,PT>& node) {
	if(isNil(node))
		return ;
	PT tmp=node.left_child;
	while(tmp!=invalidPos) {
		__data__[tmp].parent=node.parent;
		tmp=__data__[tmp].right_sibling;
	}
	if(node.left_child!=invalidPos)
		__data__[node.left_child].left_sibling=node.left_sibling;
	if(node.right_child!=invalidPos)
		__data__[node.right_child].right_sibling=node.right_sibling;
	if(node.left_sibling!=invalidPos) {
		if(node.left_child!=invalidPos)
			__data__[node.left_sibling].right_sibling=node.left_child;
		else
			__data__[node.left_sibling].right_sibling=node.right_sibling;
	}
	else {
		if(node.left_child!=invalidPos)
			__data__[node.parent].left_child=node.left_child;
		else
			__data__[node.parent].left_child=node.right_sibling;
	}
	if(node.right_sibling!=invalidPos) {
		if(node.right_child!=invalidPos)
			__data__[node.right_sibling].left_sibling=node.right_child;
		else
			__data__[node.right_sibling].left_sibling=node.left_sibling;
	}
	else {
		if(node.right_child!=invalidPos)
			__data__[node.parent].right_child=node.right_child;
		else
			__data__[node.parent].right_child=node.left_sibling;
	}
	__data__[node.self]=nilNode;
	--__size__;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::eraseBranch(const Node<K,V,PT>& node) {
	if(isNil(node))
		return ;
	if(node.left_sibling!=invalidPos)
		__data__[node.left_sibling].right_sibling=node.right_sibling;
	else
		__data__[node.parent].left_child=node.right_sibling;
	if(node.right_sibling!=invalidPos)
		__data__[node.right_sibling].left_sibling=node.left_sibling;
	else
		__data__[node.parent].right_child=node.left_sibling;
//	__data__[node.self]=nilNode;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::eraseBranch(const Node<K,V,PT>& node,size_t rsize) {
	if(isNil(node))
		return ;
	if(node.left_sibling!=invalidPos)
			__data__[node.left_sibling].right_sibling=node.right_sibling;
	else
		__data__[node.parent].left_child=node.right_sibling;
	if(node.right_sibling!=invalidPos)
			__data__[node.right_sibling].left_sibling=node.left_sibling;
	else
		__data__[node.parent].right_child=node.left_sibling;
	__data__[node.self]=nilNode;
	__size__-=rsize;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::eraseChildren(const Node<K,V,PT>& node,size_t rsize) {
	if(isNil(node))
		return ;
	__data__[node.self].left_child=invalidPos;
	__data__[node.self].right_child=invalidPos;
	__size__-=rsize;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::moveUnder(PT node,PT parent) {
	if((node>__pos__)||(parent>__pos__))
		return;
	Node<K,V,PT>& p=__data__[parent],&n=__data__[node];
	PT prc=p.right_child;
	if(prc!=invalidPos)
		__data__[prc].right_sibling=n.self;
	p.right_child=n.self;
	if(p.left_child==invalidPos)
		p.left_child=n.self;
	if(n.left_sibling!=invalidPos)
		__data__[n.left_sibling].right_sibling=n.right_sibling;
	else
		__data__[n.parent].left_child=n.right_sibling;
	if(n.right_sibling!=invalidPos)
		__data__[n.right_sibling].left_sibling=n.left_sibling;
	else
		__data__[n.parent].right_child=n.left_sibling;
	n.parent=p.self;
	n.left_sibling=prc;
	n.right_sibling=invalidPos;
}
template<typename K,typename V,typename Allocator,typename PT>
void LRTree<K,V,Allocator,PT>::swapChildren(PT nodex,PT nodey,PT parent) {
	Node<K,V,PT>& p=__data__[parent],&nx=__data__[nodex],&ny=__data__[nodey];
	if((nx.left_sibling!=invalidPos)&&(nx.left_sibling!=ny.self))
		__data__[nx.left_sibling].right_sibling=ny.self;
	if((ny.right_sibling!=invalidPos)&&(ny.right_sibling!=nx.self))
		__data__[ny.right_sibling].left_sibling=nx.self;
	if((ny.left_sibling!=invalidPos)&&(ny.left_sibling!=nx.self))
		__data__[ny.left_sibling].right_sibling=nx.self;
	if((nx.right_sibling!=invalidPos)&&(nx.right_sibling!=ny.self))
		__data__[nx.right_sibling].left_sibling=ny.self;
	if(nx.self==p.left_child)
		p.left_child=ny.self;
	else if(ny.self==p.left_child)
		p.left_child=nx.self;
	if(nx.self==p.right_child)
		p.right_child=ny.self;
	else if(ny.self==p.right_child)
		p.right_child=nx.self;
	if((nx.left_sibling!=ny.self)&&(nx.right_sibling!=ny.self)) {
		PT tmp=ny.left_sibling;
		ny.left_sibling=nx.left_sibling;
		nx.left_sibling=tmp;
		tmp=ny.right_sibling;
		ny.right_sibling=nx.right_sibling;
		nx.right_sibling=tmp;
	}
	else if(nx.right_sibling!=ny.self) {
		nx.left_sibling=ny.left_sibling;
		ny.left_sibling=nx.self;
		ny.right_sibling=nx.right_sibling;
		nx.right_sibling=ny.self;
	}
	else {
		ny.left_sibling=nx.left_sibling;
		nx.left_sibling=ny.self;
		nx.right_sibling=ny.right_sibling;
		ny.right_sibling=nx.self;
	}
}
template<typename K,typename V,typename Allocator,typename PT> template<bool check,typename allocatorType_T>
LRTree<K,V,Allocator,PT>& LRTree<K,V,Allocator,PT>::import(const LRTree<K,V,allocatorType_T,PT>& tree,StreamType stream) {
	__data__.template import<check>(tree.__data__,stream);
	__size__=tree.__size__;
	__pos__=tree.__pos__;
	return *this;
}
template<typename K,typename V,typename Allocator,typename PT> template<typename FT,typename...Args>
void LRTree<K,V,Allocator,PT>::traverseLeftRight(FT& function,const Node<K,V,PT>& node,Args... args) {
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[0];
	PT it=n.left_child;
	long depth=0;
	while(it!=invalidPos) {
		function(__data__[it],depth,args...);
		if(__data__[it].left_child!=invalidPos) {
			it=__data__[it].left_child;
			++depth;
		}
		else if(__data__[it].right_sibling!=invalidPos)
			it=__data__[it].right_sibling;
		else {
			PT ot=__data__[it].parent;
			while(ot!=invalidPos) {
				--depth;
				PT tmp=__data__[ot].right_sibling;
				if(tmp!=invalidPos) {
					ot=tmp;
					break;
				}
				else
					ot=__data__[ot].parent;
			}
			it=ot;
		}
	}
}
template<typename K,typename V,typename Allocator,typename PT> template<typename FT,typename...Args>
void LRTree<K,V,Allocator,PT>::traverseLeftRight(FT& function,const Node<K,V,PT>& node,Args... args) const {
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[__data__[0].left_child];
	PT it=n.self;
	long depth=0;
	size_t i=0;
	while((it!=invalidPos)&&((i++)<__size__)) {
		function(__data__[it],depth,args...);
		if(__data__[it].left_child!=invalidPos) {
			it=__data__[it].left_child;
			++depth;
		}
		else if(__data__[it].right_sibling!=invalidPos)
			it=__data__[it].right_sibling;
		else {
			PT ot=__data__[it].parent;
			while(ot!=invalidPos) {
				--depth;
				PT tmp=__data__[ot].right_sibling;
				if(tmp!=invalidPos) {
					ot=tmp;
					break;
				}
				else
					ot=__data__[ot].parent;
			}
			it=ot;
		}
	}
}
template<typename K,typename V,typename Allocator,typename PT> template<typename FT,typename...Args>
void LRTree<K,V,Allocator,PT>::traverseLeftRightRoot(FT& function,const Node<K,V,PT>& node,Args... args) {
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[0];
	PT it=n.self;
	long depth=0;
	while(it!=invalidPos) {
		function(__data__[it],depth,args...);
		if(__data__[it].left_child!=invalidPos) {
			it=__data__[it].left_child;
			++depth;
		}
		else if(__data__[it].right_sibling!=invalidPos)
			it=__data__[it].right_sibling;
		else {
			PT ot=__data__[it].parent;
			while(ot!=invalidPos) {
				--depth;
				PT tmp=__data__[ot].right_sibling;
				if(tmp!=invalidPos) {
					ot=tmp;
					break;
				}
				else
					ot=__data__[ot].parent;
			}
			it=ot;
		}
	}
}
template<typename K,typename V,typename Allocator,typename PT> template<typename FT,typename...Args>
void LRTree<K,V,Allocator,PT>::traverseLeftRightRoot(FT& function,const Node<K,V,PT>& node,Args... args) const {
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[0];
	PT it=n.self;
	long depth=0;
	while(it!=invalidPos) {
		function(__data__[it],depth,args...);
		if(__data__[it].left_child!=invalidPos) {
			it=__data__[it].left_child;
			++depth;
		}
		else if(__data__[it].right_sibling!=invalidPos)
			it=__data__[it].right_sibling;
		else {
			PT ot=__data__[it].parent;
			while(ot!=invalidPos) {
				--depth;
				PT tmp=__data__[ot].right_sibling;
				if(tmp!=invalidPos) {
					ot=tmp;
					break;
				}
				else
					ot=__data__[ot].parent;
			}
			it=ot;
		}
	}
}
template<typename K,typename V,typename Allocator,typename PT>  std::ostream& LRTree<K,V,Allocator,PT>::print(std::ostream& ost,const Node<K,V,PT>& node,char bbegin) const {
	auto print=[&ost,bbegin](const Node<K,V,PT>& t,long depth) {
//		if(depth>0) {
//			std::string tmp(depth,bbegin);
//			ost<<tmp<<t<<std::endl;
//		}
//		else
			ost<<t<<std::endl;
	};
	Node<K,V,PT> n=node;
	if(isNil(n))
		n=__data__[__data__[0].left_child];
	traverseLeftRight(print,n);
	return ost;
}
template <typename K,typename V,typename PT,enable_IT<is_same_CE<V,void>()> =0> std::ostream& operator<<(std::ostream& ost,const Node<K,V,PT>& node) {
	ost<<"Key: "<<node.key;
	return ost;
}
template <typename K,typename V,typename PT,enable_IT<!is_same_CE<V,void>()> =0> std::ostream& operator<<(std::ostream& ost,const Node<K,V,PT>& node) {

	ost<<node.key<<", "<<node.value;
//	ost<<"Key: "<<node.key<<"\t"<<node.value;
//	ost<<"Key: "<<node.key;
	return ost;
}
}
}
}
#endif
