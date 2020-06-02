#include "types.hh"

namespace __preprocessor__ {
for_statement_t::for_statement_t(clang_ast_node_t *self,clang_ast_node_t *declaration, clang_ast_node_t *condition,	clang_ast_node_t *step, clang_ast_node_t *block):
	self(self),declaration(declaration),condition(condition),step(step),block(block){
}
for_statement_t::for_statement_t(const clang_ast_node_t &node,const std::vector<std::string> &src):self(const_cast<clang_ast_node_t*>(&node)) {
	self=const_cast<clang_ast_node_t*>(&node);
	init(src);
}
for_statement_t::for_statement_t(const clang_ast_node_t &node):self(const_cast<clang_ast_node_t*>(&node)) {
}
void for_statement_t::init(const std::vector<std::string> &src) {
	if(self!=nullptr) {
		std::string for_source_str=__util__::replace_pattern(source(src,*self),"[[:space:]]*|for|[\\(]","");
		size_t str_pos=0,nxt_pos=0,i=0;
		nxt_pos=for_source_str.find_first_of(";",str_pos);
		if(str_pos<nxt_pos)
			declaration=(*self)[i++];
		str_pos=nxt_pos+1;
		nxt_pos=for_source_str.find_first_of(";",str_pos);
		if(str_pos<nxt_pos)
			condition=(*self)[i++];
		str_pos=nxt_pos+1;
		nxt_pos=for_source_str.find_first_of(")",str_pos);
		if(str_pos<nxt_pos)
			step=(*self)[i++];
		block=(*self)[i];
	}
}
std::string for_statement_t::begining(const std::vector<std::string> &src) {
	std::string r="";
	if(declaration!=nullptr)
		source(src,*declaration->children.front()->children.front());
	return r;
}
dependency_t::dependency_t(std::string out,std::string in,std::string condition,ptok_t type):out(out),in(in),condition(condition),type(type){
}

std::ostream& operator<<(std::ostream& ost,const for_statement_t& statement) {
	ost<<"{";
	if(statement.self!=nullptr) {
		ost<<"{"<<*(statement.self);
		if((statement.declaration!=nullptr))
			ost<<", "<<*(statement.declaration);
		else
			ost<<", ";
		if((statement.condition!=nullptr))
			ost<<", "<<*(statement.condition);
		else
			ost<<", ";
		if((statement.step!=nullptr))
			ost<<", "<<*(statement.step);
		else
			ost<<", ";
		if((statement.block!=nullptr))
			ost<<", "<<*(statement.block);
	}
	ost<<"}";
	return ost;
}
std::ostream& operator<<(std::ostream& ost,const dependency_t& dependency) {
	switch (dependency.type){
	case ptok_t::DEPENDENCY:
		ost<<dependency.type<<": "<<dependency.out<<"->"<<dependency.in;
		break;
	case ptok_t::SIMPLE_DEPENDENCY:
		ost<<dependency.type<<": "<<"self->"<<dependency.in;
		break;
	case ptok_t::CONDITIONAL_DEPENDENCY:
		ost<<dependency.type<<": "<<dependency.out<<"->"<<dependency.in<<"; if "<<dependency.condition;
		break;
	case ptok_t::SIMPLE_CONDITIONAL_DEPENDENCY:
		ost<<dependency.type<<": self->"<<dependency.in<<"; if "<<dependency.condition;
		break;
	default:
		break;
	}
	return ost;
}
std::ostream& operator<<(std::ostream &ost,variable_t type) {
	switch(type) {
	case REGULAR_VAR:
		ost<<"variable";
		break;
	case REFERENCE_VAR:
		ost<<"reference";
		break;
	case FUNCTION_VAR:
		ost<<"function parameter";
		break;
	case FOR_VAR:
		ost<<"for statement variable";
		break;
	case LAMBDA_FUNCTION:
		ost<<"lambda function";
		break;
	case LAMBDA_VAR:
		ost<<"lambda parameter";
		break;
	case NOT_VAR:
		ost<<"not a variable";
		break;
	}
	return ost;
}
}
