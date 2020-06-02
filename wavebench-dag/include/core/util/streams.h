#ifndef __STREAMS_H__
#define __STREAMS_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "stream_object.h"

typedef cudaStream_t cudaStream_t; //Typedef to stop Nsight from complaining
typedef cudaError_t cudaError_t;

namespace __core__
{

class cstreams
{
private:
	std::vector<streamObject> streams;
public:
	cstreams();
	cstreams(size_t ns,int dev=0);
	cstreams(size_t ns,int* dev);
	cstreams(const cstreams& stream)=delete;
	cstreams(cstreams&& stream);
	~cstreams();

	cstreams& operator=(const cstreams& stream)=delete;
	cstreams& operator=(cstreams&& stream);

	inline cudaStream_t& operator[](size_t i);
	inline const cudaStream_t& operator[](size_t i) const;

	inline streamObject& get(size_t i);
	inline const streamObject& get(size_t i) const;

	inline streamObject& add(int dev=0);
	inline void destroy(size_t streamId);

	inline int create(size_t ns,int dev=0);
	inline int create(size_t ns,int* dev);
	inline void destroy();
};
cstreams::cstreams(){}
cstreams::cstreams(size_t ns, int dev) {
	create(ns,dev);
}
cstreams::cstreams(size_t ns, int* dev) {
	create(ns,dev);
}

cstreams::cstreams(cstreams&& stream) {
//	destroy();
//	for(size_t i=0;i<stream.streams.size();++i)
//		streams.push_back(std::move(stream.streams[i]));
	streams=std::move();
}

cstreams::~cstreams() {
}

inline cstreams& cstreams::operator =(const cstreams& stream) {
}

inline cstreams& cstreams::operator =(cstreams&& stream) {
}

inline cudaStream_t& cstreams::operator[](size_t i) {
}

inline const cudaStream_t& cstreams::operator [](size_t i) const {
}

inline streamObject& cstreams::get(size_t i) {
}

inline const streamObject& cstreams::get(size_t i) const {
}

inline streamObject& cstreams::add(int dev) {
}

inline void cstreams::destroy(size_t streamId) {
}

inline int cstreams::create(size_t ns, int dev) {
}

inline int cstreams::create(size_t ns, int* dev) {
}

inline void cstreams::destroy() {
}
}
#endif
