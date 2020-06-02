#ifndef __PREPROCESSOR_HH__
#define __PREPROCESSOR_HH__

#include <filesystem>

#include <pragma-parser.hh>
#include "io.hh"
#include "clang/clang-wrapper.hh"
#include "preprocessor/preprocessor.hh"

namespace __preprocessor__ {
class Preprocessor {
private:
	std::filesystem::path input_file_path=std::filesystem::path();
	std::filesystem::path output_file_path=std::filesystem::path();
	std::filesystem::path tmp_file_path=std::filesystem::path();
	size_t graph_counter=0;
	void pragma_parse();
	void clang_parse(std::string options=clang_options,CXTranslationUnit_Flags flags=CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles);
	int preprocess_environment_pragmas(int offset,pragma_ast_node_t& pragma,clang_ast_node_t& block,bool timing=false,bool print=false);
	std::vector<std::string> block_preprocess(pragma_ast_node_t &pragma,clang_ast_node_t &block,const std::vector<std::string>& coarsening_opts,std::string base_identation,bool timing=false,bool print=false);
	void serial();
public:
	static const std::string clang_options;
	std::vector<std::string> input_file;
	std::vector<std::string> output_file;
	pragma_ast_t pragmaAST;
	clang_ast_t clangAST;
	std::pair<std::map<pragma_ast_node_t*,size_t>,std::vector<pragma_ast_node_t*>> pragma_translation_tables;
	std::pair<std::map<clang_ast_node_t*,size_t>,std::vector<clang_ast_node_t*>> clang_translation_tables;
	std::vector<pragma_ast_node_t*> environment_pragmas;
public:
	Preprocessor();
	~Preprocessor();
	Preprocessor(Preprocessor &&preprocessor);
	Preprocessor(const Preprocessor& preprocessor)=delete;
	Preprocessor& operator=(Preprocessor&& preprocessor);
	Preprocessor& operator=(const Preprocessor& preprocessor)=delete;

	void parse(std::string filename,std::string options=clang_options,CXTranslationUnit_Flags flags=CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles);
	void preprocess(std::string filename="",bool timing=false,bool print=false,bool serial=false);
};
}
#endif
