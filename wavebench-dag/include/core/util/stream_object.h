#ifndef __STREAM_OBJECT_H__
#define __STREAM_OBJECT_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include "../macro_definitions.h"
#include "cuda_util.h"

typedef cudaStream_t cudaStream_t; //Typedef to stop Nsight from complaining
typedef cudaError_t cudaError_t;

namespace __core__
{
class streamObject
{
private:
	cudaStream_t __stream__=0;
	int __dev__=-1;
public:
	streamObject() noexcept;
	streamObject(int dev);
	streamObject(const streamObject& stream)=delete;
	streamObject(streamObject&& stream) noexcept;
	~streamObject() noexcept;

	streamObject& operator=(const streamObject& stream)=delete;
	streamObject& operator=(streamObject&& stream) noexcept;

	inline cudaStream_t& operator*() noexcept;
	inline const cudaStream_t& operator*() const noexcept;

	inline void create(int dev=0);
	inline cudaError_t destroy() noexcept;
	inline void nullify() noexcept;

	inline void set_default() noexcept;

	inline int device() noexcept;
	inline cudaStream_t& stream() noexcept;
	inline const cudaStream_t& stream() const noexcept;

	inline cudaError_t sync() noexcept;
};

streamObject::streamObject() noexcept {}
streamObject::streamObject(int dev)
{
	create(dev);
}
streamObject::streamObject(streamObject&& stream) noexcept
{
	__stream__=stream.__stream__;
	__dev__=stream.__dev__;
	stream.nullify();
}
streamObject::~streamObject() noexcept
{
	cudaError_t ec=destroy();
	if(ec!=cudaSuccess) {
		THROW("Error at stream destruction.",STDERR_ERROR)
	}
}

streamObject& streamObject::operator=(streamObject&& stream) noexcept
{
	destroy();
	__stream__=stream.__stream__;
	__dev__=stream.__dev__;
	stream.nullify();
	return *this;
}

inline cudaStream_t& streamObject::operator*() noexcept
{
	return __stream__;
}
inline const cudaStream_t& streamObject::operator*() const noexcept
{
	return __stream__;
}

inline void streamObject::create(int dev)
{
	if(destroy()!=cudaSuccess) {
		THROW("Error at stream destruction.",STDERR_ERROR)
	}
	if((dev>=0)&&(dev!=__dev__)) dev=__get_valid_device__(dev);
	if(dev>=0)
	{
		__dev__=dev;
		if(cudaSetDevice(__dev__)!=cudaSuccess) {
			THROW("Error at device selection.")
		}
		if(cudaStreamCreate(&__stream__)!=cudaSuccess) {
			THROW("Error at device selection.")
		}
	}
}
inline cudaError_t streamObject::destroy() noexcept
{
	cudaError_t ec=cudaSuccess;
	if(__stream__!=0)
	{
		ec=cudaStreamDestroy(__stream__);
		__stream__=0;
	}
	__dev__=-1;
	return ec;
}
inline void streamObject::nullify() noexcept
{
	__stream__=0;
	__dev__=-1;
}
inline void streamObject::set_default() noexcept
{
	destroy();
}

inline int streamObject::device() noexcept
{
	return __dev__;
}
inline cudaStream_t& streamObject::stream() noexcept
{
	return __stream__;
}
inline const cudaStream_t& streamObject::stream() const noexcept
{
	return __stream__;
}
inline cudaError_t streamObject::sync() noexcept
{
	return cudaStreamSynchronize(__stream__);
}
}
#endif
