#ifndef __CORE_OMP_DAG_TIMING_HH__
#define __CORE_OMP_DAG_TIMING_HH__

#include <iostream>
#include <iomanip>

namespace __ompDAG__ {
struct ompDAGTiming_t {
	double allocation=0.;
	double creation=0.;
	double execution=0.;
	double order_allocation=0.;
	double printing=0.;
	double topological_sort=0.;
	std::ostream& report(std::ostream& ost);
};
}
#endif
