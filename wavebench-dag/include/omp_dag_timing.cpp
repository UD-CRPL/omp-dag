#include "omp_dag_timing.hh"

namespace __ompDAG__ {
std::ostream& ompDAGTiming_t::report(std::ostream& ost) {
	double total_time=allocation+creation+execution+order_allocation+topological_sort;
	if(total_time>0) {
		ost.setf( std::ios::fixed, std:: ios::floatfield );
		ost.precision(6);
		ost<<"OMP dag timing report:"<<std::endl;
		ost<<"\tGraph allocation time: "<<allocation<<" (s), as a percentage of total time: "<<allocation/total_time*100.<<"%"<<std::endl;
		ost<<"\tGraph creation time:   "<<creation<<" (s), as a percentage of total time: "<<creation/total_time*100.<<"%"<<std::endl;
		ost<<"\tGraph sorting time:    "<<topological_sort<<" (s), as a percentage of total time: "<<topological_sort/total_time*100.<<"%"<<std::endl;
		ost<<"\tTask execution time:   "<<execution<<" (s), as a percentage of total time: "<<execution/total_time*100.<<"%"<<std::endl;
		if(printing>0.)
			ost<<"\tGraph printing time:   "<<printing<<" (s)"<<std::endl;
		ost<<"\tTotal execution time:  "<<total_time<<" (s)"<<std::endl;
		ost<<resetiosflags(std::ios::floatfield);
	}
	return ost;
}
}
