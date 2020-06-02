#include "timing.h"
namespace __core__ {
namespace __debug__ {
void cpu_timer::start() {
	t1=std::chrono::high_resolution_clock::now();
}
void cpu_timer::stop() {
	t2=std::chrono::high_resolution_clock::now();
}
double cpu_timer::elapsed_time()const {
	return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count())*pow(10.,-9.);
}
std::ostream &operator<<(std::ostream &oss,const cpu_timer &timer) {
	oss<<timer.elapsed_time();
	return oss;
}
}
}
