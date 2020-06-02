#include "clang-wrapper.hh"

std::map<CXCursorKind,std::string> clangTokenNames;
std::ostream& operator<<(std::ostream& stream,CXCursorKind kind) {
	if(clangTokenNames.find(kind)!=clangTokenNames.end())
		stream<<clangTokenNames[kind];
	else
		stream<<static_cast<int>(kind);
	return stream;
}
std::ostream& operator<<(std::ostream& stream,const CXString& str) {
  stream << clang_getCString(str);
  clang_disposeString(str);
  return stream;
}
std::ostream& operator<<(std::ostream& stream,const CXCursor& cursor) {
	stream<<clang_getCursorSpelling(cursor);
	return stream;
}

namespace __clang__ {
__ast__::ASTNodeExtent clangLines(const CXCursor& cursor) {
    CXSourceRange extent           = clang_getCursorExtent( cursor );
//  CXSourceLocation startLocation = clang_getCursorLocation( cursor );
    CXSourceLocation startLocation = clang_getRangeStart( extent );
    CXSourceLocation endLocation   = clang_getRangeEnd( extent );
    unsigned int startLine = 0, startColumn = 0;
    unsigned int endLine   = 0, endColumn   = 0;
    clang_getSpellingLocation( startLocation, nullptr, &startLine, &startColumn, nullptr );
    clang_getSpellingLocation( endLocation,   nullptr, &endLine, &endColumn, nullptr );
    __ast__::ASTNodeExtent lines(startLine,startColumn,endLine,endColumn);
    return lines;
}
int getDiagnostics(CXTranslationUnit& translationUnit) {
    unsigned int ndiag = clang_getNumDiagnostics(translationUnit);
    int errors=0;
    for (unsigned int currentDiag = 0; currentDiag < ndiag; ++currentDiag) {
        CXDiagnostic diagnotic = clang_getDiagnostic(translationUnit, currentDiag);
        CXDiagnosticSeverity severity=clang_getDiagnosticSeverity(diagnotic);
        if((severity==CXDiagnostic_Error)||(severity==CXDiagnostic_Fatal)) {
        	CXString errorString = clang_formatDiagnostic(diagnotic,clang_defaultDiagnosticDisplayOptions());
        	std::cerr<<"clang error:\t"<<clang_getCString(errorString)<<std::endl;
        	clang_disposeString(errorString);
        	++errors;
        }
    }
    return errors;
}
template <typename T> CXChildVisitResult visitor(CXCursor cursor,CXCursor,CXClientData clientData) {
	if( clang_Location_isFromMainFile(clang_getCursorLocation(cursor)) == 0)
		return CXChildVisit_Continue;
	T *p=reinterpret_cast<T*>(clientData);
	clangTokenNames[clang_getCursorKind(cursor)]=toString(clang_getCursorKindSpelling(clang_getCursorKind(cursor)));
	T node=std::make_pair(p->first,&(p->first->insert(clang_getCursorKind(cursor),p->second,toString(cursor),clangLines(cursor))));
	clang_visitChildren(cursor,visitor<T>,&node);
	return CXChildVisit_Continue;
}
clang_ast_t parse(std::string filename,std::string options,CXTranslationUnit_Flags flags) {
	std::stringstream optstream(options);
	std::vector<std::string> opts;
	std::vector<const char*> copts;
	std::string tmp;
	while(getline(optstream,tmp,' '))
		opts.push_back(tmp);
	for( std::string& str : opts )
		copts.push_back(str.c_str());
	clang_ast_t ast;
	typedef std::pair<clang_ast_t*,clang_ast_node_t*> client_type;
	CXIndex index = clang_createIndex(0,0);
	CXTranslationUnit unit=nullptr;
	if(copts.size()>0)
		unit=clang_parseTranslationUnit(index,filename.c_str(),copts.data(),copts.size(),nullptr,0,flags);
	else
		unit=clang_parseTranslationUnit(index,filename.c_str(),nullptr,0,nullptr,0,flags);
	if (unit == nullptr)
		throw(std::runtime_error("Clang error: Unable to parse translation unit." ));
	int errors=getDiagnostics(unit);
	if(errors==0) {
		CXCursor cursor = clang_getTranslationUnitCursor(unit);
		client_type client_tree=std::make_pair(&ast,ast.root);
		clang_visitChildren(cursor,visitor<client_type>,&client_tree);
	}
	else
		exit(1);
	clang_disposeTranslationUnit(unit);
	clang_disposeIndex(index);
	return ast;
}
}
