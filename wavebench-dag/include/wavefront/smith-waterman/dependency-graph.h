#ifndef __DEPENDENCY_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__
#define __DEPENDENCY_GRAPH_SMITH_WATERMAN_WAVEFRONT_H__

#include "../data-structures/data-structures.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
Graph<cpuAllocator,void> smithWatermanDG(size_t n,size_t m);
}
}
}
#endif
