#ifndef __CUDA_DEBUGGERS_DEBUG_CORE_H__
#define __CUDA_DEBUGGERS_DEBUG_CORE_H__

#include "../macros/definitions.h"

#if defined(__CUDARUNTIMEQ__)
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "enum-definitions.h"
#include "debuggers.h"

namespace __core__ {
namespace __debug__ {
namespace __debug_private__ {
std::string __cuda_error_name__(const cudaError_t& error){
	return std::string(cudaGetErrorName(error));
}
template <error_ET __error_type__=USER_ERROR,errorIO_ET __error_out__=__default_error_out__,error_FT __error_format__=__default_error_message_format__,unsigned int __debug_level__=__default_debug_level__>
class __cuda_error__ {
public:
	template <error_ET error_type=__error_type__,errorIO_ET error_out=__error_out__,error_FT error_format=__error_format__,unsigned int debug_level=__debug_level__> static
	bool check(const cudaError_t& error,std::string msg="CUDA error:\t",std::string fileName="",int lineNumber=-1,std::ostream& ostr=std::cerr) {
		if(error!=cudaSuccess) {
			msg+=__cuda_error_name__(error);
			__error__<error_type,error_out,error_format,debug_level>(false,msg,fileName,lineNumber,ostr);
			return false;
		}
		return true;
	}
	bool operator()(const cudaError_t& error,std::string msg="CUDA error:\t",std::string fileName="",int lineNumber=-1,std::ostream& ostr=std::cerr) const {
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,msg,fileName,lineNumber,ostr);
	}
	template <error_ET error_type=__error_type__,errorIO_ET error_out=__error_out__,error_FT error_format=__error_format__,unsigned int debug_level=__debug_level__> static
	bool check(std::string msg="CUDA error:\t",std::string fileName="",int lineNumber=-1,std::ostream& ostr=std::cerr) {
		cudaError_t error=cudaGetLastError();
		return check<error_type,error_out,error_format,debug_level>(error,msg,fileName,lineNumber,ostr);
	}
	bool operator()(std::string msg="CUDA error:\t",std::string fileName="",int lineNumber=-1,std::ostream& ostr=std::cerr) const {
		cudaError_t error=cudaGetLastError();
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,msg,fileName,lineNumber,ostr);
	}
	bool operator=(cudaError_t&& error) const {
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,"CUDA error:\t","",-1);
	}
	bool operator=(const cudaError_t& error) const {
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,"CUDA error:\t","",-1);
	}
	bool operator<<(cudaError_t&& error) const {
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,"CUDA error:\t","",-1);
	}
	bool operator<<(const cudaError_t& error) const {
		return check<__error_type__,__error_out__,__error_format__,__debug_level__>(error,"CUDA error:\t","",-1);
	}
};
}
template <error_ET __error_type__=USER_ERROR,errorIO_ET __error_out__=__default_error_out__,error_FT __error_format__=__default_error_message_format__,unsigned int __debug_level__=__default_debug_level__>
using cudaError_CT=__debug_private__::__cuda_error__<__error_type__,__error_out__,__error_format__,__debug_level__>;
static const cudaError_CT<> cuda_checker;
}
}
#endif
#endif
