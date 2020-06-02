#ifndef __RUNTIME_WAVEFRONT_H__
#define __RUNTIME_WAVEFRONT_H__

#ifdef _OPENMP
#if _OPENMP >= 201811
#include "omp-graph.h"

namespace __core__ {
namespace __wavefront__ {
using namespace __runtime__;
}
}
#endif
#endif

#endif
