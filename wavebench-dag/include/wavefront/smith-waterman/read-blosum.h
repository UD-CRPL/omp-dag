#ifndef __READ_BLOSUM_SMITH_WATERMAN_WAVEFRONT_H__
#define __READ_BLOSUM_SMITH_WATERMAN_WAVEFRONT_H__

#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "../../linear-algebra/matrix-formats/matrix-formats.h"

namespace __core__ {
namespace __wavefront__ {
namespace __smith_waterman__ {
using namespace __linear_algebra__;
template <typename T=int> std::tuple<std::string,std::unordered_map<char,int>,Matrix<T,cpuAllocator,1>> readBlosum(std::string filename) {
	using namespace std;
	ifstream file=std::move(open_file<0>(filename));
	string line,alphabet;
	std::unordered_map<char,int> hash;
	Matrix<T,cpuAllocator,1> blosum;
	int pos=0;
	auto trim=[](string& str) { auto it = std::find_if(str.begin(),str.end(),[](char c) {return !std::isspace<char>(c,std::locale::classic());});str.erase(str.begin(), it);};
	while(getline(file,line)) {
		string tmp=line;
		tmp.erase(remove_if(tmp.begin(),tmp.end(),[](char x){return std::isspace(x);}),tmp.end());
		if(tmp.empty())
			continue;
		if(tmp[0]=='#')
			continue;
		tmp.clear();
		trim(line);
		unique_copy(line.begin(), line.end(),back_insert_iterator<string>(tmp),[](char a,char b){ return isspace(a) && isspace(b);});
	    stringstream linestream(tmp);
	    string value;
	    if(alphabet.empty()) {
	    	while(getline(linestream,value,' ')) {
	    		trim(value);
	    		hash.insert(make_pair(value.front(),pos++));
	    		alphabet.push_back(value.front());
	    	}
	    	blosum.resize(hash.size(),hash.size(),0,DEFAULT_STREAM);
	    }
	    else {
	    	pos=0;
	    	bool first=true;
	    	int j=0;
	    	while(getline(linestream,value,' ')) {
	    		trim(value);
	    		if(first) {
	    			j=hash[value.front()];
	    			first=false;
	    		}
	    		else
	    			blosum(pos++,j)=stod(value);
	    	}
	    }
	}
	close_file(file);
	return make_tuple(alphabet,hash,blosum);
}
}
}
}
#endif
