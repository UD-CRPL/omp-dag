#ifndef __CLANG_WRAPPER_H__
#define __CLANG_WRAPPER_H__

#include <iostream>
#include <clang-c/Index.h>
#include <map>
#include <utility>
#include <ast/ast.hh>

#include "../io.hh"

using namespace __ast__;

typedef ASTNode<CXCursorKind,int,std::string> clang_ast_node_t;
typedef ast_tree_t<clang_ast_node_t> clang_ast_t;
std::ostream& operator<<(std::ostream& stream,CXCursorKind kind);
std::ostream& operator<<(std::ostream& stream,const CXString& str);
std::ostream& operator<<(std::ostream& stream,const CXCursor& cursor);

namespace __clang__ {
using namespace __io__;
using ::operator<<;
__ast__::ASTNodeExtent clangLines(const CXCursor& cursor);
int getDiagnostics(CXTranslationUnit& translationUnit);
CXChildVisitResult visitor(CXCursor cursor,CXCursor,CXClientData clientData);
clang_ast_t parse(std::string filename,std::string options="-std=c++17 -I/usr/include -I/usr/local/include",CXTranslationUnit_Flags flags=CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles);
}
#endif
