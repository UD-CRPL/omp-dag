#ifndef _seq_gen_h_
#define _seq_gen_h_

#include <chrono>
#include <functional>
#include <string>
#include <random>

template <typename T=int> std::string generate_sequence(size_t size,const std::string& alphabet="ATGC") {
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

#endif
