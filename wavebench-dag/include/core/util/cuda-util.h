#ifndef __CUDA_UTIL_CORE_H__
#define __CUDA_UTIL_CORE_H__

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
#endif
#include "../macros/compiler.h"

#include "../enum-definitions.h"
#include "../debug/debug.h"

namespace __core__ {
namespace __util__ {
#ifdef __CUDARUNTIMEQ__
namespace __cuda__ {
__forceinline__  int device_count();
inline int valid_device(const int i);
inline int get_device();
__forceinline__ bool visible_devices(int first_device,int second_device);
__forceinline__ cudaPointerAttributes get_ptr_attributes(void *ptr);
int get_ptr_dev(void *ptr);
bool is_ptr_at_dev(void *ptr,int dev);
}
#else
namespace __cuda__ {
int device_count() ;
int valid_device(const int i);
int get_device() ;
bool visible_devices(int first_device,int second_device);
bool is_ptr_at_dev(void *ptr,int dev);
}
#endif
}
}
#endif
