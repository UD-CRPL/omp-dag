#ifndef __EXECUTION_POLICY_ARRAY_REDUCE_FUNCTIONAL_H__
#define __EXECUTION_POLICY_ARRAY_REDUCE_FUNCTIONAL_H__

#include "../../../types/types.h"

namespace __core__ {
namespace __functional__ {
namespace __reduce__ {
namespace __array__ {
template <bool syncpoints=false,bool contiguous=false,int nit=16,int bdim=128,int minp=bdim*8,int tnum=4> struct ReduceArrayExecutionPolicy {
	static constexpr bool syncpointsQ=syncpoints;
	static constexpr bool contiguousQ=contiguous;
	static constexpr int NIT=nit;
	static constexpr int blockdim=bdim;
	static constexpr int threadnum=tnum;
	static constexpr int minWork=minp;
};
}
}
}
}
#endif
