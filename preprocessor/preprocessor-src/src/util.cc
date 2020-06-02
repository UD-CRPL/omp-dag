#include "util.hh"
#include <iostream>
namespace __util__ {
bool pattern_match(const std::string &string, const std::string &pattern,std::regex_constants::match_flag_type match_flags,std::regex_constants::syntax_option_type regex_flags) {
	std::regex regex(pattern,regex_flags);
	return std::regex_match(string,regex,match_flags);
}
std::string replace_pattern(const std::string& string,const std::string& pattern,const std::string& format,std::regex_constants::match_flag_type match_flags,
		std::regex_constants::syntax_option_type regex_flags) {
	std::regex regex(pattern,regex_flags);
	return std::regex_replace(string,regex,format,match_flags);
}

size_t count(const std::string &string, const std::string substring) {
	size_t c=0;
	if(!substring.empty()) {
		size_t pos=string.find(substring);
		while(pos<string.size()) {
			pos=string.find(substring,pos+substring.size());
			++c;
		}
	}
	return c;
}
size_t count(const std::vector<std::string> &lines,const std::string substring) {
	size_t c=0;
	for( auto str:lines )
		c+=count(str,substring);
	return c;
}
std::string exact(const std::string &string) {
	return "\\b"+string+"\\b";
}
std::string exchange_blank_regex(const std::string &pattern) {
	std::string regex="";
	for( auto character : pattern ) {
		if(std::isblank(character))
			regex+="[[:blank:]]";
		else
			regex+=character;
	}
	return regex;
}
std::string exchange_escape_regex(const std::string &pattern) {
	std::string regex="";
	std::set<char> escape({'^', '$', '.', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|'});
	for( auto character : pattern ) {
		if(std::isblank(character))
			regex+="[[:blank:]]";
		else if(escape.count(character)==1)
			regex+="\\"+std::string(1,character);
		else
			regex+=character;
	}
	return regex;
}
}

