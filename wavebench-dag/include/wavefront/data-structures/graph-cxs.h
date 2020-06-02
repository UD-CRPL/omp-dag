#ifndef __GRAPH_CXS_DATA_STRUCTURES_WAVEFRONT_H__
#define __GRAPH_CXS_DATA_STRUCTURES_WAVEFRONT_H__

#include "../../core/core.h"
#include "../../linear-algebra/linear-algebra.h"

namespace __core__ {
namespace __wavefront__ {
namespace __data_structures__ {
template <typename VertexType,typename EdgeType,typename Allocator,typename IT=int> class GraphCXS {
private:
	Array<VertexType,Allocator> __vertices__;
	MatrixCXS<EdgeType,IT,Allocator,CRS> __graph__;
	size_t __nedges__=0;
	size_t __nvertices__=0;
	template <typename VT,typename ET,typename allocatorType_T,typename ITT> friend class GraphCXS;
public:
	GraphCXS();
	GraphCXS(GraphCXS&& graph);
	GraphCXS(const GraphCXS& graph);
	GraphCXS(size_t vertices,size_t edges,const Allocator& allocator=Allocator());
	GraphCXS(size_t vertices,size_t edges,int dev,StreamType stream=(StreamType)0,const Allocator& allocator=Allocator());
	template <typename AllocatorT> GraphCXS(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph,int dev=-1,StreamType stream=(StreamType)0,const Allocator& allocator=Allocator());
	~GraphCXS();

	GraphCXS& operator=(GraphCXS&& graph);
	GraphCXS& operator=(const GraphCXS& graph);
	template <typename AllocatorT> GraphCXS& operator=(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph);

	EdgeType& operator()(size_t i);
	const EdgeType& operator()(size_t i) const;
	VertexType& operator[](size_t i);
	const VertexType& operator[](size_t i) const;

	IT* ptr() const;
	IT* indxs() const;
	EdgeType* edges() const;
	VertexType* vertices() const;

	IT& ptr(size_t i);
	const IT& ptr(size_t i) const;
	IT& indxs(size_t i);
	const IT& indxs(size_t i) const;
	EdgeType& edges(size_t i);
	const EdgeType& edges(size_t i) const;
	VertexType& vertices(size_t i);
	const VertexType& vertices(size_t i) const;

	Array<VertexType,Allocator>& vertexArray();
	const Array<VertexType,Allocator>& vertexArray() const;
	MatrixCXS<EdgeType,IT,Allocator,CRS>& graph();
	const MatrixCXS<EdgeType,IT,Allocator,CRS>& graph() const;

	void free();

	int device() const;
	inline size_t v() const;
	inline size_t e() const;
	inline size_t nvertices() const;
	inline size_t nedges() const;

	void resize(size_t vertices,size_t edges,StreamType stream=(StreamType)0);
	void set(int v=0,StreamType stream=(StreamType)0);

	void setNZV(size_t nzv);

	template <bool check=true,typename AllocatorT=void> GraphCXS& import(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph,StreamType stream=0);
	template <bool check=true,typename AllocatorT=void> GraphCXS& import(const MatrixCXS<EdgeType,IT,AllocatorT,CRS>& graph,StreamType stream=0);
};
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS() {
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS(GraphCXS&& graph): __vertices__(std::move(graph.__vertices__)),__graph__(std::move(graph.__graph__)) {
	move(__nvertices__,graph.__nvertices__);
	move(__nedges__,graph.__nedges__);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS(const GraphCXS& graph): __vertices__(graph.__vertices__),__graph__(graph.__graph__)  {
	__nvertices__=graph.__nvertices__;
	__nedges__=graph.__nedges__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS(size_t vertices,size_t edges,const Allocator& allocator):__vertices__(Array<VertexType,Allocator>(vertices,-1,0,(StreamType)0,allocator)),
	__graph__(MatrixCXS<EdgeType,IT,Allocator,CRS>(vertices,vertices,edges,-1,(StreamType)0,allocator))  {
	__nvertices__=vertices;
	__nedges__=edges;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS(size_t vertices,size_t edges,int dev,StreamType stream,const Allocator& allocator):
__vertices__(Array<VertexType,Allocator>(vertices,dev,0,stream,allocator)),__graph__(MatrixCXS<EdgeType,IT,Allocator,CRS>(vertices,vertices,edges,dev,stream,allocator))  {
	__nvertices__=vertices;
	__nedges__=edges;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT> template<typename AllocatorT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::GraphCXS(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph,int dev,StreamType stream,const Allocator& allocator):
__vertices__(graph.__vertices__),__graph__(graph.__graph__) {
	__nvertices__=graph.__nvertices__;
	__nedges__=graph.__nedges__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>::~GraphCXS() {
	free();
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator=(GraphCXS&& graph) {
	__vertices__=std::move(graph.__vertices__);
	__graph__=std::move(graph.__graph__);
	move(__nvertices__,graph.__nvertices__);
	move(__nedges__,graph.__nedges__);
	return *this;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator=(const GraphCXS& graph) {
	return import<true>(graph,(StreamType)0);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT> template<typename AllocatorT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator=(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph) {
	return import<true>(graph,(StreamType)0);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline EdgeType& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator ()(size_t i) {
	return __graph__.values(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const EdgeType& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator ()(size_t i) const {
	return __graph__.values(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline VertexType& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator [](size_t i) {
	return __vertices__[i];
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const VertexType& GraphCXS<VertexType,EdgeType,Allocator,IT>::operator [](size_t i) const {
	return __vertices__[i];
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline IT* GraphCXS<VertexType,EdgeType,Allocator,IT>::ptr() const {
	return __graph__.ptr();
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline IT* GraphCXS<VertexType,EdgeType,Allocator,IT>::indxs() const {
	return __graph__.indxs();
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline EdgeType* GraphCXS<VertexType,EdgeType,Allocator,IT>::edges() const {
	return __graph__.values();
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline VertexType* GraphCXS<VertexType,EdgeType,Allocator,IT>::vertices() const {
	return *__vertices__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline IT& GraphCXS<VertexType,EdgeType,Allocator,IT>::ptr(size_t i) {
	return __graph__.ptr(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const IT& GraphCXS<VertexType,EdgeType,Allocator,IT>::ptr(size_t i) const {
	return __graph__.ptr(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline IT& GraphCXS<VertexType,EdgeType,Allocator,IT>::indxs(size_t i) {
	return __graph__.indxs(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const IT& GraphCXS<VertexType,EdgeType,Allocator,IT>::indxs(size_t i) const {
	return __graph__.indxs(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline EdgeType& GraphCXS<VertexType,EdgeType,Allocator,IT>::edges(size_t i) {
	return __graph__.values(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const EdgeType& GraphCXS<VertexType,EdgeType,Allocator,IT>::edges(size_t i) const {
	return __graph__.values(i);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline VertexType& GraphCXS<VertexType,EdgeType,Allocator,IT>::vertices(size_t i) {
	return __vertices__[i];
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const VertexType& GraphCXS<VertexType,EdgeType,Allocator,IT>::vertices(size_t i) const {
	return __vertices__[i];
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline Array<VertexType,Allocator>& GraphCXS<VertexType,EdgeType,Allocator,IT>::vertexArray() {
	return __vertices__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline const Array<VertexType,Allocator>& GraphCXS<VertexType,EdgeType,Allocator,IT>::vertexArray() const {
	return __vertices__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
MatrixCXS<EdgeType,IT,Allocator,CRS>& GraphCXS<VertexType,EdgeType,Allocator,IT>::graph() {
	return __graph__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
const MatrixCXS<EdgeType,IT,Allocator,CRS>& GraphCXS<VertexType,EdgeType,Allocator,IT>::graph() const {
	return __graph__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline void GraphCXS<VertexType,EdgeType,Allocator,IT>::free() {
	__vertices__.free();
	__graph__.free();
	__nvertices__=0;
	__nedges__=0;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline int GraphCXS<VertexType,EdgeType,Allocator,IT>::device() const {
	return __vertices__.device()==__graph__.device()?__graph__.device():-1;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline size_t GraphCXS<VertexType,EdgeType,Allocator,IT>::v() const {
	return __nvertices__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline size_t GraphCXS<VertexType,EdgeType,Allocator,IT>::e() const {
	return __nedges__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline size_t GraphCXS<VertexType,EdgeType,Allocator,IT>::nvertices() const {
	return __nvertices__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline size_t GraphCXS<VertexType,EdgeType,Allocator,IT>::nedges() const {
	return __nedges__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline void GraphCXS<VertexType,EdgeType,Allocator,IT>::resize(size_t vertices,size_t edges,StreamType stream) {
	__vertices__.resize(vertices,0,stream);
	__graph__.resize(vertices,vertices,edges,stream);
	__nvertices__=vertices;
	__nedges__=edges;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline void GraphCXS<VertexType,EdgeType,Allocator,IT>::set(int v,StreamType stream) {
	__vertices__.set(v,stream);
	__graph__.set(v,stream);
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT>
inline void GraphCXS<VertexType,EdgeType,Allocator,IT>::setNZV(size_t nzv) {
	__graph__.setNZV(nzv);
	__nedges__=nzv;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT> template<bool check,typename AllocatorT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>& GraphCXS<VertexType,EdgeType,Allocator,IT>::import(const GraphCXS<VertexType,EdgeType,AllocatorT,IT>& graph,StreamType stream) {
	__graph__.template import<check>(graph.__graph__,stream);
	__vertices__.template import<check>(graph.__vertices__,stream);
	__nvertices__=graph.__nvertices__;
	__nedges__=graph.__nedges__;
}
template<typename VertexType,typename EdgeType,typename Allocator,typename IT> template<bool check,typename AllocatorT>
inline GraphCXS<VertexType,EdgeType,Allocator,IT>& GraphCXS<VertexType,EdgeType,Allocator,IT>::import(const MatrixCXS<EdgeType,IT,AllocatorT,CRS>& graph,StreamType stream) {
	__graph__.template import<check>(graph,stream);
	__nvertices__=graph.n();
	__nedges__=graph.nzv();
	__vertices__.resize(__nvertices__,0,stream);
}
}
}
}
#endif
