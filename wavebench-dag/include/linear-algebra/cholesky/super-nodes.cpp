#include "super-nodes.h"

namespace __core__ {
namespace __linear_algebra__ {
namespace __cholesky__ {
std::ostream& operator<<(std::ostream& ost,const SuperNode& node){
//	ost<<node.evars<<" "<<node.parent<<" {";
//	for(size_t i=0;i<node.nodes.size();++i) {
//		if(i!=(node.nodes.size()-1))
//			ost<<node.nodes[i]<<", ";
//		else
//			ost<<node.nodes[i];
//	}
//	ost<<"}";
	ost<<node.parent<<", "<<node.time<<", "<<node.processor<<", "<<node.evars;
	return ost;
}
}
}
}
