#ifndef __STRING_GENERATOR_SMITH_WATERMAN_WAVEFRONT_H__
#define __STRING_GENERATOR_SMITH_WATERMAN_WAVEFRONT_H__

#include <chrono>
#include <random>
#include <string>
#include <unordered_map>
#include "../../core/core.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
template <typename T=int> std::string generateSequence(size_t size,const std::string& alphabet="ATGC") {
	if(alphabet.size()>0) {
		std::string result(size,' ');
		unsigned seed=std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::uniform_int_distribution<int> distribution(0,alphabet.size()-1);
		auto rand = std::bind(distribution,generator);
		for(size_t i=0;i<size;++i)
			result[i]=alphabet[rand()];
		return result;
	}
	else
		return "";
}
template <typename T=int> Array<T,cpuAllocator> translateSequence(const std::string& sequence,const std::unordered_map<char,int>& hash) {
	Array<T,cpuAllocator> s(sequence.size(),-1);
	for(size_t i=0;i<sequence.size();++i)
		s[i]=hash.at(sequence[i]);
	return s;
}
}
}
}

#endif
