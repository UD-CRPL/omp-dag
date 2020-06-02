#ifndef __OMP_GRAPH_HH__
#define __OMP_GRAPH_HH__

#include "core/core.h"
#include "linear-algebra/linear-algebra.h"
#include "wavefront/wavefront.h"
#include "third-party/graph-io/graph-io.h"
#include "omp_dag_timing.hh"

namespace __ompDAG__ {
using namespace __core__;
using namespace __wavefront__;
using namespace __third_party__::__graph_io__;
namespace __private__ {
template <typename vertex_t> struct ompDAGVertexWriter {
template <typename T> void operator()(T& graph,boost::dynamic_properties& dp) {
	dp.property("etime",boost::get(&vertex_t::etime,graph));
	dp.property("tid",boost::get(&vertex_t::tid,graph));
}
};
}
template <bool timing=false,typename vertex_t=int,typename edge_t=float,enable_IT<timing==true> = 0>
void ompDAGGraphWriter(GraphCXS<vertex_t,edge_t,cpuAllocator>& graph,std::ostream& ost) {
	__private__::ompDAGVertexWriter<vertex_t> writer;
	__third_party__::__graph_io__::write(graph,writer,ost);
}
template <bool timing=false,typename vertex_t=int,typename edge_t=float,enable_IT<timing==false> = 0>
void ompDAGGraphWriter(GraphCXS<vertex_t,edge_t,cpuAllocator>& graph,std::ostream& ost) {
	__third_party__::__graph_io__::write(graph,ost);
}
}
#endif
