#ifndef __ENUM_DEFINITIONS_H__
#define __ENUM_DEFINITIONS_H__

namespace __core__ {
enum __devices_enum__ {
	HOST,
	DEVICE,
	MANAGED
};
typedef __devices_enum__ __DeviceType__;
typedef __DeviceType__ DeviceType;

enum __memory_types_enum__ {
	NORMAL_MEM,
	PINNED_MEM,
	MANAGED_MEM
};
typedef __memory_types_enum__ __MemoryTypes__;
typedef __MemoryTypes__ MemoryType;

enum __memory_transfers_enum__ {
	CPU_CPU,
	CPU_GPU,
	CPU_MAN,
	GPU_CPU,
	GPU_GPU,
	GPU_MAN,
	MAN_CPU,
	MAN_GPU,
	MAN_MAN
};
typedef __memory_transfers_enum__ __MemoryTransferType__;
typedef __MemoryTransferType__ MemoryTransferType;

enum __sync_behavior_enum__ {
	SYNC,
	ASYNC
};
typedef __sync_behavior_enum__ __SyncBehaviorType__;
typedef __SyncBehaviorType__ SyncBehaviorType;

enum __fast_math_enum__ {
	NO_FAST_MATH,
	FAST_MATH
};
typedef __fast_math_enum__ __FastMathMode__;
typedef __FastMathMode__ FastMathMode;
#ifndef USE_FAST_MATH
#define FAST_MATH_MODE FAST_MATH
#endif
static constexpr __FastMathMode__ __default_fast_math_mode__=FAST_MATH_MODE;
}
#endif
