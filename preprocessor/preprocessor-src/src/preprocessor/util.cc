#include "util.hh"

namespace __preprocessor__ {
std::string identation(const std::string& string) {
	auto pos=string.find_first_not_of(" \t");
	if((pos!=0)&&(pos!=string.npos))
		return string.substr(0,pos);
	return "";
}
std::string identation(const std::string& prev,const std::string& prox) {
	auto prev_i=identation(prev);
	auto prox_i=identation(prox);
	return prox_i.size()>prev_i.size()?prox_i:prev_i;
}
std::string identation(const std::vector<std::string>& lines,size_t pos) {
	std::string prev="";
	std::string prox="";
	if(pos<lines.size()) {
		if(pos>0)
			prev=lines[pos-1];
		if((pos+1)<lines.size())
			prox=lines[pos+1];
		return identation(prev,prox);
	}
	return "";
}
std::string identation(const std::vector<std::string>& lines,int prev_pos,int prox_pos) {
	std::string prev="";
	std::string prox="";
	if((prev_pos<=prox_pos)&&(prox_pos>=0)) {
		if(prev_pos>0)
			prev=lines[prev_pos];
		if(prox_pos<lines.size())
			prox=lines[prox_pos];
		return identation(prev,prox);
	}
	return "";
}
std::string line_identation(const std::string &string,const std::string &identation_str) {
	return __io__::toString(line_identation(__io__::toLines(string),identation_str));
}
std::vector<std::string> line_identation(const std::vector<std::string> &lines,const std::string &identation_str) {
	std::vector<std::string> idented_lines;
	for(auto line : lines)
		idented_lines.push_back(identation_str+line);
	return idented_lines;
}
std::string remove_identation(const std::string& string,const std::string& identation_str) {
	return __io__::toString(remove_identation(__io__::toLines(string),identation_str));
}
std::string remove_single_identation(const std::string& string,const std::string& identation_str) {
	std::string result=string;
	auto pos=string.find(identation_str);
	if(pos!=std::string::npos)
		result=result.erase(pos,identation_str.size());
	return result;
}
std::vector<std::string> remove_identation(const std::vector<std::string>& lines,const std::string& identation_str) {
	std::vector<std::string> idented_lines;
	for(auto line : lines) {
		auto pos=line.find(identation_str);
		if(pos==std::string::npos)
			idented_lines.push_back(line);
		else
			idented_lines.push_back(line.erase(pos,identation_str.size()));
	}
	return idented_lines;

}
}

