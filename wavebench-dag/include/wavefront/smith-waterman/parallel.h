#ifndef __PARALLEL_SMITH_WATERMAN_WAVEFRONT_H__
#define __PARALLEL_SMITH_WATERMAN_WAVEFRONT_H__

#include "../../linear-algebra/matrix-formats/matrix-formats.h"
#include "../runtime/runtime.h"
#include "../graph-operations/graph-operations.h"
#include "super-dependency-graph.h"
#include "write-graph.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
namespace __private__ {
template <typename FT=void,typename T=void,typename ST=void>
void smithWatermanInterface(ArrayHandler<ST,size_t>& A,ArrayHandler<ST,size_t>& B,MatrixHandler<T,size_t>& M,
		GraphCXS<SmithWatermanVertex,float,cpuAllocator>& graph,std::vector<int>& order,FT& score,T gap,int threadnum=2) {
	for(size_t i=0;i<M.n();++i)
		M(i,0)=0;
	for(size_t i=0;i<M.m();++i)
		M(0,i)=0;
	auto SWWindow=[&graph,&M,&A,&B,&score,&gap](int v,int tid){
		size_t ix=graph.vertices(v).pos[0];
		size_t iy=graph.vertices(v).pos[1];
		size_t fx=graph.vertices(v).pos[0]+graph.vertices(v).dims[0];
		size_t fy=graph.vertices(v).pos[1]+graph.vertices(v).dims[1];
		for(size_t i=ix;i<fx;++i)
			for(size_t j=iy;j<fy;++j)
				M(i,j)=__max__(0,__max__(M(i-1,j-1)+score(A[i-1],B[j-1]),__max__(M(i-1,j)-gap,M(i,j-1)-gap)));
	};
	ompGraph<OMPUserOrder>(graph,SWWindow,order,threadnum);
}
template <typename FT=void,typename T=void,typename ST=void>
void smithWatermanInterfaceD(ArrayHandler<ST,size_t>& A,ArrayHandler<ST,size_t>& B,MatrixHandler<T,size_t>& M,
		GraphCXS<SmithWatermanVertexDebug,float,cpuAllocator>& graph,std::vector<int>& order,FT& score,T gap,int threadnum=2) {
	for(size_t i=0;i<M.n();++i)
		M(i,0)=0;
	for(size_t i=0;i<M.m();++i)
		M(0,i)=0;
	auto SWWindow=[&graph,&M,&A,&B,&score,&gap](int v,int tid){
		size_t ix=graph.vertices(v).pos[0];
		size_t iy=graph.vertices(v).pos[1];
		size_t fx=graph.vertices(v).pos[0]+graph.vertices(v).dims[0];
		size_t fy=graph.vertices(v).pos[1]+graph.vertices(v).dims[1];
		for(size_t i=ix;i<fx;++i)
			for(size_t j=iy;j<fy;++j)
				M(i,j)=__max__(0,__max__(M(i-1,j-1)+score(A[i-1],B[j-1]),__max__(M(i-1,j)-gap,M(i,j-1)-gap)));
	};
	ompGraph<OMPUserOrderDebug>(graph,SWWindow,order,threadnum);
}
}
template <typename FT=void,typename T=void,typename Allocator=void,uint alignment=1>
double smithWatermanP(std::string& A,std::string& B,Matrix<T,Allocator,alignment>& M,FT& score,T gap,size_t px,size_t py,int threadnum=2,bool tt=false) {
	cpu_timer timer;
	double elapsed=0;
	ArrayHandler<char,size_t>  AH,BH;
	AH.__data__=const_cast<char*>(A.c_str());
	AH.__size__=A.size();
	BH.__data__=const_cast<char*>(B.c_str());
	BH.__size__=B.size();
	M.resize(A.size()+1,B.size()+1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	if(tt)
		timer.start();
	auto dependencyGraph=smithWatermanSDG(A.size(),B.size(),px,py);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	std::vector<int> order,indegree;
	order.reserve(dependencyGraph.v());
	auto push=[&order](int v,decltype(dependencyGraph)& g){ order.push_back(v); };
	if(tt)
		timer.start();
	topologicalSort(dependencyGraph,push,indegree);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	order.pop_back();
	timer.start();
	__private__::smithWatermanInterface(AH,BH,MH,dependencyGraph,order,score,gap,threadnum);
	timer.stop();
	elapsed+=timer.elapsed_time();
	return elapsed;
}
template <typename FT=void,typename T=void,typename IT=int,typename Allocator=void,uint alignment=1>
double smithWatermanP(const Array<IT,Allocator>& A,const Array<IT,Allocator>& B,Matrix<T,Allocator,alignment>& M,FT& score,T gap,size_t px,size_t py,int threadnum=2,bool tt=false) {
	cpu_timer timer;
	double elapsed=0;
	ArrayHandler<IT,size_t>  AH(A),BH(B);
	M.resize(A.size()+1,B.size()+1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	if(tt)
		timer.start();
	auto dependencyGraph=smithWatermanSDG(A.size(),B.size(),px,py);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	std::vector<int> order,indegree;
	order.reserve(dependencyGraph.v());
	auto push=[&order](int v,decltype(dependencyGraph)& g){ order.push_back(v); };
	if(tt)
		timer.start();
	topologicalSort(dependencyGraph,push,indegree);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	order.pop_back();
	timer.start();
	__private__::smithWatermanInterface(AH,BH,MH,dependencyGraph,order,score,gap,threadnum);
	timer.stop();
	elapsed+=timer.elapsed_time();
	return elapsed;
}
template <typename FT=void,typename T=void,typename IT=int,typename Allocator=void,uint alignment=1>
double smithWatermanPD(const Array<IT,Allocator>& A,const Array<IT,Allocator>& B,Matrix<T,Allocator,alignment>& M,FT& score,T gap,size_t px,size_t py,int threadnum=2,bool tt=false) {
	cpu_timer timer;
	double elapsed=0;
	ArrayHandler<IT,size_t>  AH(A),BH(B);
	M.resize(A.size()+1,B.size()+1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	if(tt)
		timer.start();
	auto dependencyGraph=smithWatermanSDGD(A.size(),B.size(),px,py);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	std::vector<int> order,indegree;
	order.reserve(dependencyGraph.v());
	auto push=[&order](int v,decltype(dependencyGraph)& g){ order.push_back(v); };
	if(tt)
		timer.start();
	topologicalSort(dependencyGraph,push,indegree);
	if(tt) {
		timer.stop();
		elapsed+=timer.elapsed_time();
	}
	order.pop_back();
	timer.start();
	__private__::smithWatermanInterfaceD(AH,BH,MH,dependencyGraph,order,score,gap,threadnum);
	timer.stop();
	elapsed+=timer.elapsed_time();
	dependencyGraph.vertices(dependencyGraph.v()-1).etime=elapsed;
	std::ofstream file=std::move(open_file<1>("swdg.graphml"));
	writeSWD(dependencyGraph,file);
	close_file(file);
	return elapsed;
}
template <typename FT=void,typename T=void>
double smithWatermanP(std::string& A,std::string& B,FT& score,T gap,size_t px,size_t py,int threadnum=2) {
	cpu_timer timer;
	ArrayHandler<char,size_t>  AH,BH;
	AH.__data__=const_cast<char*>(A.c_str());
	AH.__size__=A.size();
	BH.__data__=const_cast<char*>(B.c_str());
	BH.__size__=B.size();
	Matrix<T,cpuAllocator,1> M(A.size()+1,B.size()+1,-1,0,DEFAULT_STREAM);
	MatrixHandler<T,size_t> MH(M);
	auto dependencyGraph=smithWatermanSDG(A.size(),B.size(),px,py);
	std::vector<int> order,indegree;
	order.reserve(dependencyGraph.v());
	auto push=[&order](int v,decltype(dependencyGraph)& g){ order.push_back(v); };
	topologicalSort(dependencyGraph,push,indegree);
	order.pop_back();
	timer.start();
	__private__::smithWatermanInterface(AH,BH,MH,dependencyGraph,order,score,gap,threadnum);
	timer.stop();
	return timer.elapsed_time();
}
}
}
}

#endif
