#ifndef __EXECUTION_POLICY_ARRAY_APPLY_FUNCTIONAL_H__
#define __EXECUTION_POLICY_ARRAY_APPLY_FUNCTIONAL_H__

#include "../../../types/types.h"

namespace __core__ {
namespace __functional__ {
namespace __apply__ {
namespace __array__ {
template<int nit=8,int bdim=256,int tnum=8,bool syncpoints=false,ReadMode rm1=read_only,ReadMode rm2=read_only,ReadMode rm3=read_only> struct ApplyArrayExecutionPolicy {
	static constexpr bool sync_points=syncpoints;
	static constexpr ReadMode read_mode=rm1;
	static constexpr ReadMode read_mode1=rm1;
	static constexpr ReadMode read_mode2=rm2;
	static constexpr ReadMode read_mode3=rm3;
	static constexpr int NIT=nit;
	static constexpr int blockdim=bdim;
	static constexpr int threadnum=tnum;
};
}
}
}
}
#endif
