#include "graph-creator.hh"

namespace __preprocessor__ {
namespace __graph__ {
std::vector<std::string> block_iteration(const std::vector<std::string> &src,const for_statement_t& statement) {
	std::string ls=source(src,*statement.declaration->children.front()->children.front());
	std::string us=source(src,*statement.condition->children[1]);
	return std::vector<std::string>({std::get<std::string>(statement.declaration->children.front()->value),ls,us,"(("+us+")-("+ls+"))"});
}
}
}
