#ifndef __ASTEXTENT_H__
#define __ASTEXTENT_H__

#include <functional>
#include <iostream>
namespace __ast__ {
struct ASTNodeLinePosition {
	int line=-1;
	int column=-1;
	ASTNodeLinePosition(int l=-1,int c=-1);
	ASTNodeLinePosition(const ASTNodeLinePosition& pos);
	ASTNodeLinePosition(ASTNodeLinePosition&& pos);
	template <typename T> ASTNodeLinePosition(const T& pos) {
		set(pos);
	}
	~ASTNodeLinePosition();
	ASTNodeLinePosition& operator=(const ASTNodeLinePosition& pos);
	ASTNodeLinePosition& operator=(ASTNodeLinePosition&& pos);
	template <typename T> ASTNodeLinePosition& operator=(const T& pos) {
		set(pos);
		return *this;
	}
	void set(int l,int c);
	template <typename T> void set(const T& pos) {
		line=pos.line;
		column=pos.column;
	}
	void clear();
	std::ostream& print(std::ostream& ost,const std::string& begin="[",const std::string& end="]",const std::string& sep=", ") const;
};
std::ostream& operator<<(std::ostream& ost,const ASTNodeLinePosition& pos);

struct ASTNodeExtent {
	ASTNodeLinePosition begin=ASTNodeLinePosition();
	ASTNodeLinePosition end=ASTNodeLinePosition();
	ASTNodeExtent(int bl=-1,int bc=-1,int el=-1,int ec=-1);
	ASTNodeExtent(const ASTNodeExtent& extent);
	ASTNodeExtent(ASTNodeExtent&& extent);
	template <typename T> ASTNodeExtent(const T& extent): begin(extent.begin),end(extent.end) {
	}
	template <typename T> ASTNodeExtent(const T& b,const T& e): begin(b),end(e) {
	}
	~ASTNodeExtent();
	ASTNodeExtent& operator=(const ASTNodeExtent& extent);
	ASTNodeExtent& operator=(ASTNodeExtent&& extent);
	template <typename T> ASTNodeExtent& operator=(const T& extent) {
		set(extent);
		return *this;
	}
	void set(int fl,int fc,int ll,int lc);
	template <typename T> void set(const T& b,const T& e) {
		begin.set(b);
		end.set(e);
	}
	template <typename T> void set(const T& extent) {
		begin.set(extent.begin);
		end.set(extent.end);
	}
	void clear();
	std::ostream& print(std::ostream& ost,std::function<void(std::ostream&,const ASTNodeExtent&)> printer=[](std::ostream& o,const ASTNodeExtent& extent) { o<<extent.begin; } ) const;
};
std::ostream& operator<<(std::ostream& ost,const ASTNodeExtent& extent);
}
#endif
