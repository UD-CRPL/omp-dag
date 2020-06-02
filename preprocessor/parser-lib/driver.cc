#include "driver.hh"
#include "parser.hh"

Driver::Driver() {
}
Driver::~Driver() {
	file="";
	ast.clear();
}
int Driver::parse(const std::string& f) {
  file = f;
  location.initialize(&file);
  scanBegin();
  yy::parser parse(*this);
  parse.set_debug_level(trace_parsing);
  int res = parse();
  scanEnd();
  return res;
}
