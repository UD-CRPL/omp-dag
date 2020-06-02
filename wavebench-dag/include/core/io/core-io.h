#ifndef __CORE_IO_H__
#define __CORE_IO_H__

#include <iostream>
#include <string>

#include "../debug/debug.h"
#include "../data-structures/data-structures.h"
#include "../types/vector-types.h"

namespace __core__ {
namespace __io__ {
namespace __io_core__ {
template <typename T,typename U=T,typename Allocator=void> std::ostream& print(const Array<T,Allocator>& array,std::size_t size,std::ostream &ost=std::cerr,std::string separator=", ",std::string begin="{",std::string end="}") {
	size=std::min(size,array.size());
	ost<<begin;
    for(size_t i=0;i<size;++i) {
    	if(i<size-1)
    		ost<<static_cast<U>(array[i])<<separator;
    	else
            ost<<static_cast<U>(array[i]);
    }
    ost<<end;
    return ost;
}
template <typename T,typename Allocator=void> std::ostream & operator<<(std::ostream &ost,const Array<T,Allocator>& array) {
	return print(array,array.size(),ost);
}
template <typename T,typename U=T,int n=0,uint alignment=0> std::ostream& print(const __core__::__type__::Vector<T,n,alignment>& v,std::ostream &ost=std::cerr,std::string separator=", ",std::string begin="{",std::string end="}") {
	ost<<begin;
    for(size_t i=0;i<n;++i) {
    	if(i<n-1)
    		ost<<static_cast<U>(v[i])<<separator;
    	else
            ost<<static_cast<U>(v[i]);
    }
    ost<<end;
    return ost;
}
template <typename T,int n,uint alignment> std::ostream& operator<<(std::ostream &ost,const __core__::__type__::Vector<T,n,alignment>& v) {
	return print(v,ost);
}
}
}
}
#endif
