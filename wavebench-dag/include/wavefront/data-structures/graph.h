#ifndef __GRAPH_DATA_STRUCTURES_H__
#define __GRAPH_DATA_STRUCTURES_H__

#include "../../core/core.h"

namespace __core__ {
namespace __wavefront__ {
namespace __data_structures__ {
template <typename V,typename PT> struct EdgeNode {
	PT svertex;
	PT dvertex;
	PT self;
	PT previous;
	PT next;
	V value;
	EdgeNode(PT sv,PT dv,PT s,PT p,PT n): svertex(sv),dvertex(dv),self(s),previous(p),next(n),value(V()){
	}
	EdgeNode(PT sv,PT dv,PT s,PT p,PT n,const V& v): svertex(sv),dvertex(dv),self(s),previous(p),next(n),value(v){
	}
};
template <typename PT> struct EdgeNode<void,PT> {
	PT svertex;
	PT dvertex;
	PT self;
	PT previous;
	PT next;
	EdgeNode(PT sv,PT dv,PT s,PT p,PT n): svertex(sv),dvertex(dv),self(s),previous(p),next(n){
	}
};
template <typename V,typename PT> std::ostream& operator<<(std::ostream& ost,const EdgeNode<V,PT>& edge){
	ost<<edge.dvertex<<" "<<edge.value;
	return ost;
}
template <typename PT> std::ostream& operator<<(std::ostream& ost,const EdgeNode<void,PT>& edge){
	ost<<edge.dvertex;
	return ost;
}
template <typename AT,typename PT=int,typename IT=int> struct Vertex {
	AT attributes;
	PT self;
	PT head;
	PT tail;
	IT size;
	Vertex(PT s,PT h,PT t,IT sz=0): attributes(AT()),self(s),head(h),tail(t),size(sz){
	}
	Vertex(const AT& at,PT s,PT h,PT t,IT sz): attributes(at),self(s),head(h),tail(t),size(sz){
	}
};
template <typename PT,typename IT> struct Vertex<void,PT,IT> {
	PT self;
	PT head;
	PT tail;
	IT size;
	Vertex(PT s,PT h,PT t,IT sz=0): self(s),head(h),tail(t),size(sz){
	}
};
template <typename AT,typename PT,typename IT> std::ostream& operator<<(std::ostream& ost,const Vertex<AT,PT,IT>& vertex){
	ost<<vertex.self<<" "<<vertex.attributes;
	return ost;
}
template <typename PT,typename IT> std::ostream& operator<<(std::ostream& ost,const Vertex<void,PT,IT>& vertex){
	ost<<vertex.self;
	return ost;
}

template <typename Allocator,typename AT,typename V=void,typename PT=int,typename IT=int> class Graph {
public:
	typedef Vertex<AT,PT,IT> VertexType;
	typedef EdgeNode<V,PT> EdgeNodeType;
	static constexpr PT invalidPos=is_signed_CE<PT>()?-1:__numeric_limits__<PT>::max;
private:
	Array<Vertex<AT,PT,IT>,Allocator> __vertices__;
	Array<EdgeNode<V,PT>,Allocator> __edges__;
	size_t __vertices_size__=0;
	size_t __edges_size__=0;
	size_t __vertices_pos__=0;
	size_t __edges_pos__=0;
	void checkEdgesSpace(size_t add);
	void checkVerticesSpace(size_t add);
public:
	Graph();
	Graph(Graph&& graph);
	Graph(const Graph& graph);
	Graph(size_t vertices,int dev=-1,const Allocator& allocator=Allocator());
	Graph(size_t vertices,size_t edges,int dev,StreamType stream,const Allocator& allocator=Allocator());
	template <typename Allocator_T> Graph(const Graph<Allocator_T,AT,V,PT,IT>& graph,int dev=-1,StreamType stream=0,const Allocator& allocator=Allocator());
	~Graph();

	Graph& operator=(Graph&& graph);
	Graph& operator=(const Graph& graph);
	template <typename Allocator_T> Graph& operator=(const Graph<Allocator_T,AT,V,PT,IT>& graph);

	inline Vertex<AT,PT,IT>& operator[](size_t i);
	inline const Vertex<AT,PT,IT>& operator[](size_t i) const;

	void free();
	void clear();

	inline size_t n() const;
	inline size_t m() const;
	inline size_t vertex_quantity() const;
	inline size_t edge_quantity() const;

	inline Vertex<AT,PT,IT>& vertex(size_t i);
	inline const Vertex<AT,PT,IT>& vertex(size_t i) const;
	inline EdgeNode<V,PT>& edge(size_t i);
	inline const EdgeNode<V,PT>& edge(size_t i) const;

	inline Array<Vertex<AT,PT,IT>,Allocator>& vertices();
	inline const Array<Vertex<AT,PT,IT>,Allocator>& vertices() const;
	inline Array<EdgeNode<V,PT>,Allocator>& edges();
	inline const Array<EdgeNode<V,PT>,Allocator>& edges() const;

	void reserveVertices(size_t vertices,StreamType stream=0);
	void reserveEdges(size_t edges,StreamType stream=0);
	void reserve(size_t vertices,size_t edges,StreamType stream=0);

	void resizeVertices(size_t vertices,StreamType stream=0);
	void resizeEdges(size_t edges,StreamType stream=0);
	void resize(size_t vertices,size_t edges,StreamType stream=0);

	template <typename AAT=AT,enable_IT<is_same_CE<AAT,void>()> = 0> Vertex<AT,PT,IT>& insertVertex();
	template <typename AAT=AT,enable_IT<!is_same_CE<AAT,void>()> = 0> Vertex<AT,PT,IT>& insertVertex(const AAT& attributes=AAT());
	template <typename AVT=V,enable_IT<is_same_CE<AVT,void>()> = 0> EdgeNode<V,PT>& insertEdge(PT vid,PT uid);
	template <typename AVT=V,enable_IT<!is_same_CE<AVT,void>()> = 0> EdgeNode<V,PT>& insertEdge(PT vid,PT uid,const AVT& value=AVT());
	template <typename AVT=V,enable_IT<is_same_CE<AVT,void>()> = 0> EdgeNode<V,PT>& insertEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u);
	template <typename AVT=V,enable_IT<!is_same_CE<AVT,void>()> = 0> EdgeNode<V,PT>& insertEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u,const AVT& value=AVT());

	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0>
	void initVertices(size_t pos=0,size_t len=__numeric_limits__<size_t>::max);
	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0>
	void initEdges(size_t pos=0,size_t len=__numeric_limits__<size_t>::max);

	void eraseVertex(PT vid);
	void eraseVertex(const Vertex<AT,PT,IT>& v);
	void eraseEdge(PT eid);
	void eraseEdge(const EdgeNode<V,PT>& e);
	void eraseEdge(PT vid,PT uid);
	void eraseEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u);

	template <typename Allocator_T> Graph& import(const Graph<Allocator_T,AT,V,PT,IT>& graph,StreamType stream=0);

	template <typename allocatorType_T=Allocator,enable_IT<eq_CE(allocatorType_T::location,HOST)> = 0>
	std::ostream& print(std::ostream &ost=std::cerr,std::string separator=" ",std::string begin=" ",std::string end="",std::string newRow="\n") const;
};
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::Graph() {
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::Graph(Graph&& graph): __vertices__(std::move(graph.__vertices__)),__edges__(std::move(graph.__edges__)) {
	move(__vertices_size__,graph.__vertices_size__);
	move(__edges_size__,graph.__edges_size__);
	move(__vertices_pos__,graph.__vertices_pos__);
	move(__edges_pos__,graph.__edges_pos__);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::Graph(const Graph& graph): __vertices__(graph.__vertices__),__edges__(graph.__edges__) {
	__vertices_size__=graph.__vertices_size__;
	__edges_size__=graph.__edges_size__;
	__vertices_pos__=graph.__vertices_pos__;
	__edges_pos__=graph.__edges_pos__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::Graph(size_t vertices,int dev,const Allocator& allocator): __vertices__(Array<Vertex<AT,PT,IT>,Allocator>(vertices,dev,allocator)),
__edges__(Array<EdgeNode<V,PT>,Allocator>(vertices*2,dev,allocator)) {
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::Graph(size_t vertices,size_t edges,int dev,StreamType stream,const Allocator& allocator): __vertices__(Array<Vertex<AT,PT,IT>,Allocator>(vertices,dev,0,stream,allocator)),
__edges__(Array<EdgeNode<V,PT>,Allocator>(edges,dev,0,stream,allocator)) {
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename Allocator_T>
Graph<Allocator,AT,V,PT,IT>::Graph(const Graph<Allocator_T,AT,V,PT,IT>& graph,int dev,StreamType stream,const Allocator& allocator):
	__vertices__(graph.__vertices__,dev,stream,allocator),__edges__(graph.__edges__,dev,stream,allocator) {
	__vertices_size__=graph.__vertices_size__;
	__edges_size__=graph.__edges_size__;
	__vertices_pos__=graph.__vertices_pos__;
	__edges_pos__=graph.__edges_pos__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>::~Graph() {
	free();
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>& Graph<Allocator,AT,V,PT,IT>::operator=(Graph&& graph) {
	__vertices__=std::move(graph.__vertices__);
	__edges__=std::move(graph.__edges__);
	move(__vertices_size__,graph.__vertices_size__);
	move(__edges_size__,graph.__edges_size__);
	move(__vertices_pos__,graph.__vertices_pos__);
	move(__edges_pos__,graph.__edges_pos__);
	return *this;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Graph<Allocator,AT,V,PT,IT>& Graph<Allocator,AT,V,PT,IT>::operator=(const Graph& graph) {
	return import(graph);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename Allocator_T>
Graph<Allocator,AT,V,PT,IT>& Graph<Allocator,AT,V,PT,IT>::operator=(const Graph<Allocator_T,AT,V,PT,IT>& graph) {
	return import(graph);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::operator [](size_t i) {
	return __vertices__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
const Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::operator [](size_t i) const {
	return __vertices__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::checkEdgesSpace(size_t add) {
	if((__edges_pos__+add)>__edges__.capacity())
		reserveEdges(((__edges_pos__+add+1)*3)/2);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::checkVerticesSpace(size_t add) {
	if((__vertices_pos__+add)>__vertices__.capacity())
		reserveVertices(((__vertices_pos__+add+1)*3)/2);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::free() {
	__edges__.free();
	__vertices__.free();
	__vertices_size__=0;
	__edges_size__=0;
	__vertices_pos__=0;
	__edges_pos__=0;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::clear() {
	__edges__.clear();
	__vertices__.clear();
	__vertices_size__=0;
	__edges_size__=0;
	__vertices_pos__=0;
	__edges_pos__=0;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
size_t Graph<Allocator,AT,V,PT,IT>::n() const {
	return __vertices_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
size_t Graph<Allocator,AT,V,PT,IT>::m() const {
	return __edges_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
size_t Graph<Allocator,AT,V,PT,IT>::vertex_quantity() const {
	return __vertices_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
size_t Graph<Allocator,AT,V,PT,IT>::edge_quantity() const {
	return __edges_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::vertex(size_t i) {
	return __vertices__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
const Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::vertex(size_t i) const {
	return __vertices__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::edge(size_t i) {
	return __edges__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
const EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::edge(size_t i) const {
	return __edges__[i];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Array<Vertex<AT,PT,IT>,Allocator>& Graph<Allocator,AT,V,PT,IT>::vertices() {
	return __vertices__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
const Array<Vertex<AT,PT,IT>,Allocator>& Graph<Allocator,AT,V,PT,IT>::vertices() const {
	return __vertices__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
Array<EdgeNode<V,PT>,Allocator>& Graph<Allocator,AT,V,PT,IT>::edges() {
	return __edges__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
const Array<EdgeNode<V,PT>,Allocator>& Graph<Allocator,AT,V,PT,IT>::edges() const {
	return __edges__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::reserveVertices(size_t vertices,StreamType stream) {
	__vertices__.resize(vertices,0,stream);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::reserveEdges(size_t edges,StreamType stream) {
	__edges__.resize(edges,0,stream);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::reserve(size_t vertices,size_t edges,StreamType stream) {
	reserveVertices(vertices,stream);
	reserveEdges(edges,stream);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::resizeVertices(size_t vertices,StreamType stream) {
	__vertices__.resize(vertices,0,stream);
	__vertices_size__=vertices;
	__vertices_pos__=__vertices_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::resizeEdges(size_t edges,StreamType stream) {
	__edges__.resize(edges,0,stream);
	__edges_size__=edges;
	__edges_pos__=__edges_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::resize(size_t vertices,size_t edges,StreamType stream) {
	resizeVertices(vertices,stream);
	resizeEdges(edges,stream);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename Allocator_T>
Graph<Allocator,AT,V,PT,IT>& Graph<Allocator,AT,V,PT,IT>::import(const Graph<Allocator_T,AT,V,PT,IT>& graph,StreamType stream) {
	__vertices__.import(graph.__vertices__,stream);
	__edges__.import(graph.__edges__,stream);
	__vertices_size__=graph.__vertices_size__;
	__edges_size__=graph.__edges_size__;
	__vertices_pos__=graph.__vertices_pos__;
	__edges_pos__=graph.__edges_pos__;
	return *this;
}

template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
void Graph<Allocator,AT,V,PT,IT>::initVertices(size_t pos,size_t len) {
	len=__min__(len,__vertices__.capacity());
	for(size_t i=pos;i<len;++i)
		__vertices__[i]=Vertex<AT,PT,IT>(i,invalidPos,invalidPos,0);
	__vertices_size__=len;
	__vertices_pos__=len;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
void Graph<Allocator,AT,V,PT,IT>::initEdges(size_t pos,size_t len) {
	len=__min__(len,__edges__.capacity());
	for(size_t i=pos;i<len;++i)
		__edges__[i]=EdgeNode<V,PT>(invalidPos,invalidPos,invalidPos,invalidPos,invalidPos);
}

template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AAT,enable_IT<is_same_CE<AAT,void>()>>
Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::insertVertex() {
	checkVerticesSpace(1);
	__vertices__[__vertices_pos__]=Vertex<AT,PT,IT>(__vertices_pos__,invalidPos,invalidPos,0);
	++__vertices_size__;
	++__vertices_pos__;
	return __vertices__[__vertices_pos__-1];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AAT,enable_IT<!is_same_CE<AAT,void>()>>
Vertex<AT,PT,IT>& Graph<Allocator,AT,V,PT,IT>::insertVertex(const AAT& attributes) {
	checkVerticesSpace(1);
	__vertices__[__vertices_pos__]=Vertex<AT,PT,IT>(attributes,__vertices_pos__,invalidPos,invalidPos,0);
	++__vertices_size__;
	++__vertices_pos__;
	return __vertices__[__vertices_pos__-1];
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AVT,enable_IT<is_same_CE<AVT,void>()>>
EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::insertEdge(PT vid,PT uid) {
	checkEdgesSpace(1);
	EdgeNode<V,PT>& edge=__edges__[__edges_pos__]=EdgeNode<V,PT>(vid,uid,__edges_pos__,invalidPos,invalidPos);
	++__edges_size__;
	++__edges_pos__;
	if(__vertices__[vid].size!=0) {
		__edges__[__vertices__[vid].tail].next=edge.self;
		edge.previous=__edges__[__vertices__[vid].tail].self;
		__vertices__[vid].tail=edge.self;
	}
	else {
		__vertices__[vid].head=edge.self;
		__vertices__[vid].tail=edge.self;
	}
	__vertices__[vid].size+=1;
	return edge;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AVT,enable_IT<!is_same_CE<AVT,void>()>>
EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::insertEdge(PT vid,PT uid,const AVT& value) {
	checkEdgesSpace(1);
	EdgeNode<V,PT>& edge=__edges__[__edges_pos__]=EdgeNode<V,PT>(vid,uid,value,__edges_pos__,invalidPos,invalidPos);
	++__edges_size__;
	++__edges_pos__;
	if(__vertices__[vid].size!=0) {
		__edges__[__vertices__[vid].tail].next=edge.self;
		edge.previous=__edges__[__vertices__[vid].tail].self;
		__vertices__[vid].tail=edge.self;
	}
	else {
		__vertices__[vid].head=edge.self;
		__vertices__[vid].tail=edge.self;
	}
	__vertices__[vid].size+=1;
	return edge;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AVT,enable_IT<is_same_CE<AVT,void>()>>
EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::insertEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u) {
	return insertEdge(v.self,u.self);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename AVT,enable_IT<!is_same_CE<AVT,void>()>>
EdgeNode<V,PT>& Graph<Allocator,AT,V,PT,IT>::insertEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u,const AVT& value) {
	return insertEdge(v.self,u.self,value);
}

template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseVertex(PT vid) {
	if(__vertices__[vid].size>0)
		__edges_size__=__edges_size__>__vertices__[vid].size?__edges_size__-__vertices__[vid].size:0;
	__vertices__[vid]=VertexType(invalidPos,invalidPos,invalidPos,0);
	if(__vertices_size__>0)
		--__vertices_size__;
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseVertex(const Vertex<AT,PT,IT>& v) {
	eraseVertex(v.self);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseEdge(PT eid) {
	if(eid!=invalidPos) {
		EdgeNode<V,PT>& edge=__edges__[eid];
		if(edge.svertex!=invalidPos) {
			Vertex<AT,PT,IT>& vertex=__vertices__[edge.svertex];
			if(edge.previous!=invalidPos)
				__edges__[edge.previous].next=edge.next;
			else
				vertex.head=edge.next;
			if(edge.next!=invalidPos)
				__edges__[edge.next].previous=edge.previous;
			else
				vertex.tail=edge.previous;
			if(vertex.size>0)
				vertex.size-=1;
		}
		if(__edges_size__>0)
			--__edges_size__;
	}
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseEdge(const EdgeNode<V,PT>& e) {
	eraseEdge(e.self);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseEdge(PT vid,PT uid) {
	if(vid!=invalidPos) {
		PT it=__vertices__[vid].head;
		while(it!=invalidPos) {
			if(__edges__[it].dvertex==uid) {
				eraseEdge(__edges__[it].self);
				break;
			}
			it=__edges__[it].next;
		}
	}
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT>
void Graph<Allocator,AT,V,PT,IT>::eraseEdge(const Vertex<AT,PT,IT>& v,const Vertex<AT,PT,IT>& u) {
	eraseEdge(v.self,u.self);
}
template <typename Allocator,typename AT,typename V,typename PT,typename IT> template <typename allocatorType_T,enable_IT<eq_CE(allocatorType_T::location,HOST)>>
std::ostream& Graph<Allocator,AT,V,PT,IT>::print(std::ostream &ost,std::string separator,std::string begin,std::string end,std::string newRow) const {
	ost<<n()<<" "<<m()<<std::endl;
	for(size_t i=0;i<__vertices_pos__;++i) {
		Vertex<AT,PT,IT> vertex=__vertices__[i];
		if(vertex.self!=invalidPos) {
			ost<<vertex;
			if(vertex.size!=0) {
				ost<<begin;
				PT it=vertex.head;
				while(it!=invalidPos) {
					if(it==vertex.tail)
						ost<<__edges__[it];
					else
						ost<<__edges__[it]<<separator;
					it=__edges__[it].next;
				}
				ost<<end;
			}
			ost<<newRow;
		}
	}
	return ost;
}
}
}
}
#endif
