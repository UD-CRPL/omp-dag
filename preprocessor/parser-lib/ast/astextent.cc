#include "astextent.hh"

namespace __ast__ {
ASTNodeLinePosition::ASTNodeLinePosition(int l,int c) {
	set(l,c);
}
ASTNodeLinePosition::ASTNodeLinePosition(const ASTNodeLinePosition& pos) {
	set(pos);
}
ASTNodeLinePosition::ASTNodeLinePosition(ASTNodeLinePosition&& pos) {
	set(pos);
	pos.line;
	pos.column;
}
ASTNodeLinePosition::~ASTNodeLinePosition() {
	clear();
}
ASTNodeLinePosition& ASTNodeLinePosition::operator=(const ASTNodeLinePosition& pos) {
	set(pos);
	return *this;
}
ASTNodeLinePosition& ASTNodeLinePosition::operator=(ASTNodeLinePosition&& pos) {
	set(pos);
	pos.line;
	pos.column;
	return *this;
}
void ASTNodeLinePosition::set(int l,int c) {
	line=l;
	column=c;
}
void ASTNodeLinePosition::clear() {
	set(-1,-1);
}
std::ostream& ASTNodeLinePosition::print(std::ostream& ost,const std::string& begin,const std::string& end,const std::string& sep) const {
	ost<<begin<<line<<sep<<column<<end;
	return ost;
}

std::ostream& operator<<(std::ostream& ost,const ASTNodeLinePosition& pos) {
	return pos.print(ost);
}

ASTNodeExtent::ASTNodeExtent(int bl,int bc,int el,int ec): begin(ASTNodeLinePosition(bl,bc)), end(ASTNodeLinePosition(el,ec)) {
}
ASTNodeExtent::ASTNodeExtent(const ASTNodeExtent& extent): begin(extent.begin),end(extent.end) {
}
ASTNodeExtent::ASTNodeExtent(ASTNodeExtent&& extent): begin(std::move(extent.begin)),end(std::move(extent.end)) {
}
ASTNodeExtent::~ASTNodeExtent(){
	clear();
}
ASTNodeExtent& ASTNodeExtent::operator=(const ASTNodeExtent& extent) {
	set(extent);
	return *this;
}
ASTNodeExtent& ASTNodeExtent::operator=(ASTNodeExtent&& extent) {
	begin=std::move(extent.begin);
	end=std::move(extent.end);
	return *this;
}
void ASTNodeExtent::set(int fl,int fc,int ll,int lc) {
	begin.set(fl,fc);
	begin.set(ll,lc);
}
void ASTNodeExtent::clear() {
	begin.clear();
	end.clear();
}
std::ostream& ASTNodeExtent::print(std::ostream& ost,std::function<void(std::ostream&,const ASTNodeExtent&)> printer) const {
	printer(ost,*this);
	return ost;
}
std::ostream& operator<<(std::ostream& ost,const ASTNodeExtent& extent) {
	return extent.print(ost);
}
}
