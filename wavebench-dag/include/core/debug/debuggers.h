#ifndef __DEBUGGERS_DEBUG_CORE_H__
#define __DEBUGGERS_DEBUG_CORE_H__

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "enum-definitions.h"
#include "../macros/compiler.h"

namespace __core__ {
namespace __debug__ {
namespace __debug_private__ {
template <error_ET error_type,unsigned int debug_level> __attribute__((optimize(0)))
constexpr bool __error_printer_activator__(){
    return ((RUNTIME_ERROR==error_type)&&((debug_level&runtime_debug)==runtime_debug))||
           ((MEMORY_ERROR==error_type)&&((debug_level&memory_debug)==memory_debug))||
           ((API_ERROR==error_type)&&((debug_level&api_debug)==api_debug))||
           ((KERNEL_LAUNCH_ERROR==error_type)&&((debug_level&kernel_launch_debug)==kernel_launch_debug))||
           ((USER_ERROR==error_type)&&((debug_level&user_debug)==user_debug))||
           ((WARNING_ERROR==error_type)&&((debug_level&warning_debug)==warning_debug))||
           (ALWAYS_ERROR==error_type);
}
template <errorIO_ET error_out=__default_error_out__,error_FT error_format=__default_error_message_format__> __forceinline__  __attribute__((optimize(2)))
void __error_printer__(const std::string& message,const std::string& fileName,int lineNumber,std::ostream& ostr=std::cerr) {
	std::string error_message;
	std::string file_info="";
	switch(error_format){
	case long_error_message:
		if(!(fileName.empty()||lineNumber<0))
			file_info="\n\tFile name:\n\t\t"+fileName+"\n\tLine number:\n\t\t"+std::to_string(lineNumber);
		if(message.empty())
			error_message="\n*********************************************************************\n\n\tThere has been an error"+file_info+
			"\n\n*********************************************************************";
		else
			error_message="\n*********************************************************************\n\n\tThere has been an error, the error was:\n\t\t"
					+message+file_info+"\n\n*********************************************************************";
		break;
	case short_error_message:
		if(!(fileName.empty()||lineNumber<0))
			file_info="\tFN: "+fileName+"\tLN: "+std::to_string(lineNumber);
		error_message=message+file_info;
		break;
	case minimal_error_message:
		if(!(fileName.empty()||lineNumber<0))
			file_info=", "+fileName+", "+std::to_string(lineNumber);
		error_message=message+file_info;
		break;
	case only_msg_error_message:
		error_message=message;
	break;
	}
	switch(error_out){
	case stderr_error:
		std::cerr<<error_message<<std::endl;
		break;
	case stdout_error:
		std::cout<<error_message<<std::endl;
		break;
	case throw_error:
		throw(std::runtime_error(error_message));
		break;
	case assert_error:
		std::cerr<<error_message<<std::endl;
#ifdef NDEBUG
#undef NDEBUG
		assert(0);
#define NDEBUG
#else
		assert(0);
#endif
		break;
	case file_out_error:
		ostr<<error_message<<std::endl;
		break;
	}
}
template <error_ET error_type=USER_ERROR,errorIO_ET error_out=__default_error_out__,error_FT error_format=__default_error_message_format__,unsigned int debug_level=__default_debug_level__> __forceinline__ __attribute__((optimize(2))) __attribute__((flatten))
typename std::enable_if<__error_printer_activator__<error_type,debug_level>()==true,void>::type __error__(bool condition,const std::string& message,const std::string& fileName,int lineNumber,std::ostream& ostr=std::cerr) {
	if(!condition) {
		switch(error_type){
		case NO_ERROR:
			break;
		case RUNTIME_ERROR:
			if((debug_level&runtime_debug)==runtime_debug)
				__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		case MEMORY_ERROR:
			if((debug_level&memory_debug)==memory_debug)
				__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		case API_ERROR:
			if((debug_level&api_debug)==api_debug)
				__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		case KERNEL_LAUNCH_ERROR:
			if((debug_level&kernel_launch_debug)==kernel_launch_debug)
				__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		case USER_ERROR:
			if((debug_level&user_debug)==user_debug)
				__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		case WARNING_ERROR:
			if((debug_level&warning_debug)==warning_debug)
				__error_printer__<stderr_error,error_format>(message,fileName,lineNumber);
			break;
		case ALWAYS_ERROR:
			__error_printer__<error_out,error_format>(message,fileName,lineNumber,ostr);
			break;
		}
	}
}
template <error_ET error_type=USER_ERROR,errorIO_ET error_out=__default_error_out__,error_FT error_format=__default_error_message_format__,unsigned int debug_level=__default_debug_level__> __forceinline__ __attribute__((optimize(2))) __attribute__((flatten))
typename std::enable_if<__error_printer_activator__<error_type,debug_level>()==false,void>::type __error__(bool condition,const std::string& message,const std::string& fileName,int lineNumber,std::ostream& ostr=std::cerr) {
}
}
}
}
#endif
