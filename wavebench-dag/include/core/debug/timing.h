#ifndef __TIMING_DEBUG_CORE_H__
#define __TIMING_DEBUG_CORE_H__

#include <chrono>
#include <cmath>
#include <iostream>

#include "../macros/definitions.h"
#ifdef __CUDARUNTIMEQ__
#include <cuda_runtime.h>
#endif

namespace __core__ {
namespace __debug__ {
struct cpu_timer {
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
	void start();
	void stop();
	double elapsed_time()const;
};
#if defined(__CUDARUNTIMEQ__)
struct gpu_timer {
	cudaEvent_t t1;
	cudaEvent_t t2;
	cudaStream_t stream=0;
	gpu_timer() {
		cudaEventCreate(&t1);
		cudaEventCreate(&t2);
	}
	~gpu_timer() {
		cudaEventDestroy(t1);
		cudaEventDestroy(t2);
	}
	__inline__ __attribute__((always_inline)) void start() {
		cudaDeviceSynchronize();
		cudaEventRecord(t1,stream);
	}
	__inline__ __attribute__((always_inline)) void stop() {
		cudaEventRecord(t2,stream);
		cudaEventSynchronize(t2);
	}
	__inline__ __attribute__((always_inline)) double elapsed_time() const {
		float elapsed;
		cudaEventElapsedTime(&elapsed,t1,t2);
		return ((double)elapsed)*pow(10,-3);
	}
};
std::ostream &operator<<(std::ostream &oss,const gpu_timer &timer) {
	oss<<timer.elapsed_time();
	return oss;
}

template <typename T> __inline__ __attribute__((always_inline)) T& operator<<(T &t,const gpu_timer &timer) {
	t=timer.elapsed_time();
	return t;
}
#else
typedef cpu_timer gpu_timer;
#endif
std::ostream &operator<<(std::ostream &oss,const cpu_timer &timer);
template <typename T> __inline__ __attribute__((always_inline)) T& operator<<(T &t,const cpu_timer &timer) {
	t=timer.elapsed_time();
	return t;
}
}
}
#endif
 
