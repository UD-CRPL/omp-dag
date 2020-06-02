#ifndef __ENUM_DEFINITIONS_TYPE_CORE_H__
#define __ENUM_DEFINITIONS_TYPE_CORE_H__

namespace __core__ {
namespace __type__ {
enum __read_modes_enum__ {
	read_only,
	normal_read
};
typedef __read_modes_enum__ __ReadMode__;
typedef __ReadMode__ ReadMode;

enum __rounding_modes_enum__ {
	RN,
	RD,
	RU,
	RZ,
	RDFLT
};
typedef __rounding_modes_enum__ __RoundingMode__;
typedef __RoundingMode__ RoundingMode;
#ifndef ROUNDING_MODE
#define ROUNDING_MODE RN
#endif
static constexpr __RoundingMode__ __default_rounding_mode__=ROUNDING_MODE;
}
}
#endif
