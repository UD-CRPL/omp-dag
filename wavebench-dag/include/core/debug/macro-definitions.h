#ifndef __MACRO_DEFINITIONS_DEBUG_CORE_H__
#define __MACRO_DEFINITIONS_DEBUG_CORE_H__

#define __CHOOSE_MACRO_DEBUG__(_0,_1,_2,_3,_4,_5,_6, name, ...) name
#define __BAD_ARG_COUNT_DEBUG_MACRO__(...) static_assert(false,"Bad quantity of arguments in macro, line: "+to_string(__LINE__)+", in file: "+__FILE__)

#define __COMMAQ_DEBUG__(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,0,0,0,0,0,0,1)
#define __TRIGGER_PARENTHESIS_DEBUG__(...) ,

#define __IS_NOT_EMPTY_DEBUG__(...) __IS_NOT_EMPTY_HELPER_DEBUG__( __COMMAQ_DEBUG__(__TRIGGER_PARENTHESIS_DEBUG__ __VA_ARGS__ ), __COMMAQ_DEBUG__(__TRIGGER_PARENTHESIS_DEBUG__ __VA_ARGS__ ( )))

#define __INVOKE_PASTE_3__(x,y,z) __PASTE_3_DEBUG__(x,y,z)
#define __PASTE_3_DEBUG__(_0, _1, _2) _0 ## _1 ## _2
#define __IS_NOT_EMPTY_HELPER_DEBUG__(_0,_1) __COMMAQ_DEBUG__(__PASTE_3_DEBUG__(__NOT_EMPTY_DEBUG__, _0,_1))
#define __NOT_EMPTY_DEBUG__10 ,

#ifndef NDEBUG
#define __ERROR_PRINTER_7__(condition,message,error_type,out_type,file,frmt_type,level_type) __debug__::__debug_private__::template __error__<error_type,out_type,frmt_type,level_type>(condition,message,__FILE__,__LINE__,file)
#define __ERROR_PRINTER_6__(condition,message,error_type,out_type,file,frmt_type) __debug__::__debug_private__::template __error__<error_type,out_type,frmt_type>(condition,message,__FILE__,__LINE__,file)
#define __ERROR_PRINTER_5__(condition,message,error_type,out_type,file) __debug__::__debug_private__::template __error__<error_type,out_type>(condition,message,__FILE__,__LINE__,file)
#define __ERROR_PRINTER_4__(condition,message,error_type,out_type) __debug__::__debug_private__::template __error__<error_type,out_type>(condition,message,__FILE__,__LINE__)
#define __ERROR_PRINTER_3__(condition,message,error_type) __debug__::__debug_private__::template __error__<error_type>(condition,message,__FILE__,__LINE__)
#define __ERROR_PRINTER_2__(condition,message) __debug__::__debug_private__::__error__(condition,message,__FILE__,__LINE__)
#define __ERROR_PRINTER_1__(condition) __debug__::__debug_private__::__error__(condition,"",__FILE__,__LINE__)
#define __ERROR_PRINTER_0__(...) __debug__::__debug_private__::__error__(0,"",__FILE__,__LINE__)
#define __ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __ERROR_PRINTER__(...) __ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__ERROR_PRINTER_7__,__ERROR_PRINTER_6__,__ERROR_PRINTER_5__,__ERROR_PRINTER_4__,__ERROR_PRINTER_3__,__ERROR_PRINTER_2__,__ERROR_PRINTER__)(__VA_ARGS__)

#if defined(__CUDARUNTIMEQ__)
#define __CUDA_ERROR_PRINTER_6__(error_code,error_type,out_type,file,frmt_type,level_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type,frmt_type,level_type>::check(error_code,"CUDA error:\t",__FILE__,__LINE__,file)
#define __CUDA_ERROR_PRINTER_5__(error_code,error_type,out_type,file,frmt_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type,frmt_type>::check(error_code,"CUDA error:\t",__FILE__,__LINE__,file)
#define __CUDA_ERROR_PRINTER_4__(error_code,error_type,out_type,file) __debug__::__debug_private__::__cuda_error__<error_type,out_type>::check(error_code,"CUDA error:\t",__FILE__,__LINE__,file)
#define __CUDA_ERROR_PRINTER_3__(error_code,error_type,out_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type>::check(error_code,"CUDA error:\t",__FILE__,__LINE__)
#define __CUDA_ERROR_PRINTER_2__(error_code,error_type) __debug__::__debug_private__::__cuda_error__<error_type>::check(error_code,"CUDA error:\t",__FILE__,__LINE__)
#define __CUDA_ERROR_PRINTER_1__(error_code) __debug__::__debug_private__::__cuda_error__<>::check(error_code,"CUDA error:\t",__FILE__,__LINE__)
#define __CUDA_ERROR_PRINTER_0__(...) __debug__::__debug_private__::__cuda_error__<>::check("CUDA error:\t",__FILE__,__LINE__)
#define __CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CUDA_ERROR_PRINTER__(...) __CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CUDA_ERROR_PRINTER_6__,__CUDA_ERROR_PRINTER_5__,__CUDA_ERROR_PRINTER_4__,__CUDA_ERROR_PRINTER_3__,__CUDA_ERROR_PRINTER_2__,__CUDA_ERROR_PRINTER__)(__VA_ARGS__)

#define __CHECK_CUDA_ERROR_PRINTER_5__(error_type,out_type,file,frmt_type,level_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type,frmt_type,level_type>::check("CUDA error:\t",__FILE__,__LINE__,file)
#define __CHECK_CUDA_ERROR_PRINTER_4__(error_type,out_type,file,frmt_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type,frmt_type>::check("CUDA error:\t",__FILE__,__LINE__,file)
#define __CHECK_CUDA_ERROR_PRINTER_3__(error_type,out_type,file) __debug__::__debug_private__::__cuda_error__<error_type,out_type>::check("CUDA error:\t",__FILE__,__LINE__,file)
#define __CHECK_CUDA_ERROR_PRINTER_2__(error_type,out_type) __debug__::__debug_private__::__cuda_error__<error_type,out_type>::check("CUDA error:\t",__FILE__,__LINE__)
#define __CHECK_CUDA_ERROR_PRINTER_1__(error_type) __debug__::__debug_private__::__cuda_error__<error_type>::check("CUDA error:\t",__FILE__,__LINE__)
#define __CHECK_CUDA_ERROR_PRINTER_0__(...) __debug__::__debug_private__::__cuda_error__<>::check("CUDA error:\t",__FILE__,__LINE__)
#define __CHECK_CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CHECK_CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CHECK_CUDA_ERROR_PRINTER__(...) __CHECK_CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define check_cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CHECK_CUDA_ERROR_PRINTER_5__,__CHECK_CUDA_ERROR_PRINTER_4__,__CHECK_CUDA_ERROR_PRINTER_3__,__CHECK_CUDA_ERROR_PRINTER_2__,__CHECK_CUDA_ERROR_PRINTER__)(__VA_ARGS__)
#else
#define __CUDA_ERROR_PRINTER_1__(error_code,...) ((void)0)
#define __CUDA_ERROR_PRINTER_0__(...) ((void)0)
#define __CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CUDA_ERROR_PRINTER__(...) __CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER__)(__VA_ARGS__)

#define __CHECK_CUDA_ERROR_PRINTER_1__(...) ((void)0)
#define __CHECK_CUDA_ERROR_PRINTER_0__(...) ((void)0)
#define __CHECK_CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CHECK_CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CHECK_CUDA_ERROR_PRINTER__(...) __CHECK_CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define check_cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER__)(__VA_ARGS__)
#endif

#else
#define __ERROR_PRINTER_1__(condition,...) condition
#define __ERROR_PRINTER_0__(...) ((void)0)
#define __ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __ERROR_PRINTER__(...) __ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__ERROR_PRINTER_1__,__ERROR_PRINTER_1__,__ERROR_PRINTER_1__,__ERROR_PRINTER_1__,__ERROR_PRINTER_1__,__ERROR_PRINTER_1__,__ERROR_PRINTER__)(__VA_ARGS__)

#define __CUDA_ERROR_PRINTER_1__(error_code,...) error_code
#define __CUDA_ERROR_PRINTER_0__(...) ((void)0)
#define __CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CUDA_ERROR_PRINTER__(...) __CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER_1__,__CUDA_ERROR_PRINTER__)(__VA_ARGS__)

#define __CHECK_CUDA_ERROR_PRINTER_1__(...) ((void)0)
#define __CHECK_CUDA_ERROR_PRINTER_0__(...) ((void)0)
#define __CHECK_CUDA_ERROR_PRINTER_HELPER__(type,...) __INVOKE_PASTE_3__(__CHECK_CUDA_ERROR_PRINTER_,type,__)(__VA_ARGS__)
#define __CHECK_CUDA_ERROR_PRINTER__(...) __CHECK_CUDA_ERROR_PRINTER_HELPER__(__IS_NOT_EMPTY_DEBUG__(__VA_ARGS__),__VA_ARGS__)
#define check_cuda_error(...) __CHOOSE_MACRO_DEBUG__(__VA_ARGS__,__BAD_ARG_COUNT_DEBUG_MACRO__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER_1__,__CHECK_CUDA_ERROR_PRINTER__)(__VA_ARGS__)
#endif
#endif
