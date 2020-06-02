#include "cuda-util.h"

namespace __core__ {
namespace __util__ {
#ifdef __CUDARUNTIMEQ__
namespace __cuda__ {
__forceinline__  int device_count() {
	int i=0;
	cuda_error(cudaGetDeviceCount(&i),API_ERROR);
	return i;
}
inline int valid_device(const int i) {
	if(i<0)
		return 0;
	else if(i<device_count())
		return i;
	return 0;
}
inline int get_device() {
	int i=0;
	cuda_error(cudaGetDevice(&i),API_ERROR);
	return i;
}
__forceinline__ bool visible_devices(int first_device,int second_device) {
	if(first_device==second_device)
		return true;
	int is_visible=0;
	cuda_error(cudaDeviceCanAccessPeer(&is_visible,first_device,second_device),API_ERROR);
	return is_visible==1;
}
__forceinline__ cudaPointerAttributes get_ptr_attributes(void *ptr) {
	cudaPointerAttributes R;
	cuda_error(cudaPointerGetAttributes(&R,ptr),API_ERROR);
	return R;
}
int get_ptr_dev(void *ptr) {
	cudaPointerAttributes R;
	cuda_error(cudaPointerGetAttributes(&R,ptr),API_ERROR);
	return R.device;
}
bool is_ptr_at_dev(void *ptr,int dev) {
	cudaPointerAttributes R;
	cuda_error(cudaPointerGetAttributes(&R,ptr),API_ERROR);
	return dev==R.device;
}
}
#else
namespace __cuda__ {
int device_count() {
	return 0;
}
int valid_device(const int i) {
	return 0;
}
int get_device() {
	return 0;
}
bool visible_devices(int first_device,int second_device) {
	return false;
}
bool is_ptr_at_dev(void *ptr,int dev) {
	return 0;
}
}
#endif
}
}
