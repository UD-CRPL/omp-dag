#ifndef __ENUM_DEFINITIONS_DEBUG_CORE_H__
#define __ENUM_DEFINITIONS_DEBUG_CORE_H__

namespace __core__ {
namespace __debug__ {
enum __error_types_enum__ : int {
	NO_ERROR=0,
	RUNTIME_ERROR,
	MEMORY_ERROR,
	API_ERROR,
	KERNEL_LAUNCH_ERROR,
	USER_ERROR,
	WARNING_ERROR,
	ALWAYS_ERROR=-1
};
typedef __error_types_enum__ __error_type__;
typedef __error_type__ error_ET;

enum __debug_levels_enum__ : unsigned int{
	no_debug=0,
	runtime_debug=0x1,
	memory_debug=0x2,
	api_debug=0x4,
	kernel_launch_debug=0x8,
	last_debug_flag=kernel_launch_debug,
	user_debug=(last_debug_flag*2),
	warning_debug=(last_debug_flag*4),
	all_debug=(last_debug_flag*8)-1,
	debug_all=all_debug,
	__NDEBUG__=no_debug
};
typedef __debug_levels_enum__ __debug_levels_type__;
typedef __debug_levels_type__ debugLevels_ET;
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL all_debug
#endif
static constexpr unsigned int __default_debug_level__=static_cast<unsigned int>(DEBUG_LEVEL);

enum __error_io_enum__ {
	stderr_error,
	stdout_error,
	throw_error,
	assert_error,
	file_out_error
};
typedef __error_io_enum__ __error_io_type__;
typedef __error_io_type__ errorIO_ET;
#ifndef DEBUG_ERROR_TYPE
#define DEBUG_ERROR_TYPE throw_error
#endif
static constexpr errorIO_ET __default_error_out__=DEBUG_ERROR_TYPE;

enum __error_message_format_enum__ {
	long_error_message,
	short_error_message,
	minimal_error_message,
	only_msg_error_message
};
typedef __error_message_format_enum__ __error_message_format_type__;
typedef __error_message_format_type__ error_FT;
#ifndef DEBUG_MESSAGE_FORMAT
#define DEBUG_MESSAGE_FORMAT long_error_message
#endif
static constexpr error_FT __default_error_message_format__=DEBUG_MESSAGE_FORMAT;
}
}
#endif
