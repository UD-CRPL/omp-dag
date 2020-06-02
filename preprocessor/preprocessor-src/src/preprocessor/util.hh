#ifndef __PREPROCESSOR_UTIL_HH__
#define __PREPROCESSOR_UTIL_HH__

#include <string>
#include <vector>

#include <ast/ast.hh>
#include "../io.hh"

namespace __preprocessor__ {
template <typename T,typename T0,typename...TN> std::string source(const std::vector<std::string>& src_code,const __ast__::ASTNode<T,T0,TN...>& node) {
	auto bounded=[](int p,int l,int u) { return ((l<=p)&&(p<u)); };
	std::string result="";
	const __ast__::ASTNodeExtent& extent=node.extent;
	if(bounded(extent.begin.line-1,0,src_code.size())&&bounded(extent.end.line-1,0,src_code.size())&&(extent.begin.line<=extent.end.line)) {
		if(extent.begin.line!=extent.end.line) {
			for(size_t i=static_cast<size_t>(extent.begin.line-1);i<=static_cast<size_t>(extent.end.line-1);++i) {
				if(i==static_cast<size_t>(extent.begin.line-1))
					result+=src_code[i].substr(extent.begin.column-1)+"\n";
				else if(i==static_cast<size_t>(extent.end.line-1))
					result+=src_code[i].substr(0,extent.end.column)+"\n";
				else
					result+=src_code[i]+"\n";
			}
		}
		else
			result=src_code[extent.begin.line-1].substr(extent.begin.column-1,extent.end.column-extent.begin.column);
	}
	return result;
}
std::string identation(const std::string& string);
std::string identation(const std::string& prev,const std::string& prox);
std::string identation(const std::vector<std::string>& lines,size_t pos);
std::string identation(const std::vector<std::string>& lines,int prev_pos,int prox_pos);
std::string line_identation(const std::string& string,const std::string& identation_str);
std::vector<std::string> line_identation(const std::vector<std::string>& lines,const std::string& identation_str);
std::string remove_identation(const std::string& string,const std::string& identation_str);
std::string remove_single_identation(const std::string& string,const std::string& identation_str);
std::vector<std::string> remove_identation(const std::vector<std::string>& lines,const std::string& identation_str);
}
#endif
