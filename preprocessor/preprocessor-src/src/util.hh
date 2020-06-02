#ifndef __UTIL_HH__
#define __UTIL_HH__

#include <iterator>
#include <set>
#include <string>
#include <regex>
#include <utility>
#include <vector>

#include <ast/astextent.hh>

namespace __util__ {
template <typename T> void insert(std::vector<T>& v,typename std::vector<T>::iterator it,std::vector<T>&& u) {
	v.insert(it,u.begin(),u.end());
}
template <typename T> void insert(std::vector<T>& v,size_t start,std::vector<T>&& u) {
	v.insert(v.begin()+start,u.begin(),u.end());
}
template <typename T> void insert(std::vector<T>& v,std::vector<T>&& u) {
	v.insert(v.end(),u.begin(),u.end());
}

enum class regex_mode_e {
	STRINGS=1,
	POSITIONS=2,
	STRING_POSITIONS=3,
	INPLACE,
	OFFPLACE
};
using regex_mode=regex_mode_e;

template <regex_mode mode=regex_mode::STRINGS,typename IT=int,std::enable_if_t<mode==regex_mode::STRINGS,int> = 0>
std::vector<std::string> patterns(const std::string& string,const std::string& pattern,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::vector<std::string> matches;
	std::regex regex(pattern,regex_flags);
	auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
	auto matches_end = std::sregex_iterator();
	for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
		std::smatch match = *i;
		matches.push_back(match.str());
	}
	return matches;
}
template <regex_mode mode=regex_mode::STRINGS,typename IT=int,std::enable_if_t<mode==regex_mode::POSITIONS,int> = 0>
std::vector<std::pair<IT,IT>> patterns(const std::string& string,const std::string& pattern,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::vector<std::pair<IT,IT>> matches;
	std::regex regex(pattern,regex_flags);
	auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
	auto matches_end = std::sregex_iterator();
	for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
		std::smatch match = *i;
		matches.push_back(std::pair<IT,IT>(match.position(),match.str().size()));
	}
	return matches;
}
template <regex_mode mode=regex_mode::STRINGS,typename IT=int,std::enable_if_t<mode==regex_mode::STRING_POSITIONS,int> = 0>
std::vector<std::pair<std::string,IT>> patterns(const std::string& string,const std::string& pattern,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::vector<std::pair<std::string,IT>> matches;
	std::regex regex(pattern,regex_flags);
	auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
	auto matches_end = std::sregex_iterator();
	for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
		std::smatch match = *i;
		matches.push_back(std::pair<std::string,int>(match.str(),match.position()));
	}
	return matches;
}

template <regex_mode mode=regex_mode::STRINGS,std::enable_if_t<mode==regex_mode::STRINGS,int> = 0>
std::vector<std::string> patterns(const std::vector<std::string>& lines,const std::string& pattern,
		std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::vector<std::string> matches;
	std::regex regex(pattern,regex_flags);
	for(auto string : lines) {
		auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
		auto matches_end = std::sregex_iterator();
		for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
			std::smatch match = *i;
			matches.push_back(match.str());
		}
	}
	return matches;
}
template <regex_mode mode=regex_mode::STRINGS,std::enable_if_t<mode==regex_mode::STRING_POSITIONS,int> = 0>
std::vector<std::pair<std::string,__ast__::ASTNodeLinePosition>> patterns(const std::vector<std::string>& lines,const std::string& pattern,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
		std::vector<std::pair<std::string,__ast__::ASTNodeLinePosition>> matches;
		std::regex regex(pattern,regex_flags);
		int l=1;
		for(auto string : lines) {
			auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
			auto matches_end = std::sregex_iterator();
			for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
				std::smatch match = *i;
				matches.push_back(std::pair<std::string,__ast__::ASTNodeLinePosition>(match.str(),__ast__::ASTNodeLinePosition(l,match.position())));
			}
			++l;
		}
		return matches;
}
template <regex_mode mode=regex_mode::STRINGS,std::enable_if_t<mode==regex_mode::POSITIONS,int> = 0>
std::vector<std::pair<__ast__::ASTNodeLinePosition,int>> patterns(const std::vector<std::string>& lines,const std::string& pattern,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
		std::vector<std::pair<__ast__::ASTNodeLinePosition,int>> matches;
		std::regex regex(pattern,regex_flags);
		int l=1;
		for(auto string : lines) {
			auto matches_begin=std::sregex_iterator(string.begin(),string.end(),regex,match_flags);
			auto matches_end = std::sregex_iterator();
			for (std::sregex_iterator i=matches_begin; i!=matches_end; ++i) {
				std::smatch match = *i;
				matches.push_back(std::pair<__ast__::ASTNodeLinePosition,int>(__ast__::ASTNodeLinePosition(l,match.position()+1),match.str().size()));
			}
			++l;
		}
		return matches;
}

bool pattern_match(const std::string& string,const std::string& pattern,
		std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript);

std::string replace_pattern(const std::string& string,const std::string& pattern,const std::string& format,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript);
template <regex_mode mode=regex_mode::OFFPLACE,std::enable_if_t<mode==regex_mode::OFFPLACE,int> = 0>
std::vector<std::string> replace_pattern(const std::vector<std::string>& lines,const std::string& pattern,const std::string& format,
		std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::vector<std::string> replaced;
	std::regex regex(pattern,regex_flags);
	for(auto string : lines)
		replaced.push_back(std::regex_replace(string,regex,format,match_flags));
	return replaced;
}
template <regex_mode mode=regex_mode::OFFPLACE,std::enable_if_t<mode==regex_mode::INPLACE,int> = 0>
void replace_pattern(std::vector<std::string>& lines,const std::string& pattern,const std::string& format,std::regex_constants::match_flag_type match_flags = std::regex_constants::match_default,
		std::regex_constants::syntax_option_type regex_flags=std::regex_constants::ECMAScript) {
	std::regex regex(pattern,regex_flags);
	for(size_t i=0;i<lines.size();++i)
		lines[i]=std::regex_replace(lines[i],regex,format,match_flags);
}
size_t count(const std::string& string,const std::string substring);
size_t count(const std::vector<std::string>& lines,const std::string substring);

std::string exact(const std::string& string);

const std::string cxx_identifier_regex="[_[:alpha:]][_[:alnum:]]*";
const std::string int_regex="[[:digit:]]+";
const std::string boolean_operators_regex="[<>]|<=|>=|==|&&|\\|\\||!";

std::string exchange_blank_regex(const std::string& pattern);
std::string exchange_escape_regex(const std::string& pattern);

}
#endif
