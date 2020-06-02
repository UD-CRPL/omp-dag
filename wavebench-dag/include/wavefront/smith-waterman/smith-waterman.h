#ifndef __SMITH_WATERMAN_WAVEFRONT_H__
#define __SMITH_WATERMAN_WAVEFRONT_H__

#include "serial.h"
#include "dependency-graph.h"
#include "string-generator.h"
#include "super-dependency-graph.h"
#ifdef _OPENMP
#if _OPENMP >= 201811
#include "parallel.h"
#endif
#endif
#include "read-blosum.h"
#include "write-graph.h"

namespace __core__ {
namespace __wavefront__ {
using namespace __smith_waterman__;
}
}
#endif
