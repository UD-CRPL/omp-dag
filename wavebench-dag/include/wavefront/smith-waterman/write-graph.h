#ifndef __WRITE_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__
#define __WRITE_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__

#include "super-dependency-graph.h"
#include "../../third-party/graph-io/graph-io.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
namespace __private__ {
struct VertexWriter {
template <typename T> void operator()(T& graph,boost::dynamic_properties& dp) {
	dp.property("etime",boost::get(&SmithWatermanVertexDebug::etime,graph));
	dp.property("tid",boost::get(&SmithWatermanVertexDebug::tid,graph));
}
};
}
template <typename T=int> void writeSWD(GraphCXS<SmithWatermanVertexDebug,float,cpuAllocator>& graph,std::ostream& ost) {
	__private__::VertexWriter writer;
	__third_party__::__graph_io__::write(graph,writer,ost);
}
}
}
}
#endif
