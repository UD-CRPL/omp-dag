#ifndef __FUNCTIONS_MACROS_CORE_H__
#define __FUNCTIONS_MACROS_CORE_H__

#include "compiler.h"

#define STRIP_PARENTHESIS_H(...) __VA_ARGS__
#define INVOKE(expr) expr
#define STRIP_PARENTHESIS(args) INVOKE(STRIP_PARENTHESIS_H args)

#define CHOOSE_MACRO(_0,_1,_2,_3,_4,_5,_6,_7, name, ...) name
#define BAD_ARG_COUNT(...) static_assert(false,"Bad quantity of arguments in macro, line: "+to_string(__LINE__)+", in file: "+__FILE__)
#define JOIN(_0,_1) _0##_1

#define UFOR_WA(counter_type,counter_name,begin,end,arguments_types,arguments_types_and_names,arguments,function_body) { typedef struct { \
	static inline __attribute__((always_inline)) __attribute__((optimize(3))) __host_device__ void fn(counter_type counter_name,STRIP_PARENTHESIS(arguments_types_and_names)){\
	STRIP_PARENTHESIS(function_body)} \
	} unrolled_fn;\
	const_for<unrolled_fn,counter_type,begin,end-1>::template iterator<STRIP_PARENTHESIS(arguments_types)>(STRIP_PARENTHESIS(arguments)); }

#define UFOR_NA(counter_type,counter_name,begin,end,function_body) { typedef struct { \
	static inline __attribute__((always_inline)) __attribute__((optimize(3))) __host_device__ void fn(counter_type counter_name){\
	STRIP_PARENTHESIS(function_body)} \
	} unrolled_fn;\
	const_for<unrolled_fn,counter_type,begin,end-1>::iterator(); }

#define UFOR(...) CHOOSE_MACRO(__VA_ARGS__,UFOR_WA,BAD_ARG_COUNT,BAD_ARG_COUNT,UFOR_NA,BAD_ARG_COUNT,BAD_ARG_COUNT,BAD_ARG_COUNT)(__VA_ARGS__)

#define __macro_for_WA__(counter_type,counter_name,begin,end,arguments_types,arguments_types_and_names,arguments,function_body) \
	for(counter_type counter_name=(counter_type)begin;counter_name<(counter_type)end;++counter_name) { \
	STRIP_PARENTHESIS(function_body) }
#define __macro_for_NA__(counter_type,counter_name,begin,end,function_body) \
	for(counter_type counter_name=(counter_type)begin;counter_name<(counter_type)end;++counter_name) { \
	STRIP_PARENTHESIS(function_body) }

#define MACRO_FOR(...) CHOOSE_MACRO(__VA_ARGS__,__macro_for_WA__,BAD_ARG_COUNT,BAD_ARG_COUNT,__macro_for_NA__,BAD_ARG_COUNT,BAD_ARG_COUNT,BAD_ARG_COUNT)(__VA_ARGS__)

#define __unroll_meta__(...) UFOR(__VA_ARGS__)
#define META_UNROLL
#ifdef META_UNROLL
#define __unroll_gpu__(...) UFOR(__VA_ARGS__)
#define __unroll_cpu__(...) UFOR(__VA_ARGS__)
#else
#ifdef PARTIAL_META_UNROLL
#define __unroll_gpu__(...) MACRO_FOR(__VA_ARGS__)
#define __unroll_cpu__(...) UFOR(__VA_ARGS__)
#else
#define __unroll_gpu__(...)	MACRO_FOR(__VA_ARGS__)
#define __unroll_cpu__(...) MACRO_FOR(__VA_ARGS__)
#endif
#endif

#endif
