#ifndef __ASTNODE_HH__
#define __ASTNODE_HH__

#include <functional>
#include <iostream>
#include <variant>
#include <type_traits>
#include <utility>
#include <vector>

#include "astextent.hh"

template <typename T0, typename ... Ts> std::ostream& operator<<(std::ostream& ost,const std::variant<T0, Ts...>& v) {
    std::visit([&](auto && arg){ ost << arg;}, v);
    return ost;
}

namespace __ast__ {
using ::operator<<;
template <typename...T> struct is_same_type_t;
template <typename U,typename V,typename...T> struct is_same_type_t<U,V,T...> {
	static constexpr bool value=std::is_same_v<U,V>&&is_same_type_t<V,T...>::value;
};
template <typename U> struct is_same_type_t<U> {
	static constexpr bool value=true;
};
template <> struct is_same_type_t<> {
	static constexpr bool value=false;
};
template <typename...T> inline constexpr bool is_same_type=is_same_type_t<T...>::value;
template <typename...T> inline constexpr bool is_same_type_b=is_same_type_t<std::remove_reference_t<std::remove_cv_t<T>>...>::value;

template <typename TType,typename T0,typename...T> struct ASTNode {
    typedef TType token_type;
    typedef std::variant<T0,T...> variant_type;
    TType token=token_type();
    ASTNode* parent=nullptr;
    std::vector<ASTNode*> children=std::vector<ASTNode*>();
    std::variant<T0,T...> value=0;
    ASTNodeExtent extent=ASTNodeExtent();
    ASTNode(){}
    ASTNode(ASTNode&& node): token(node.token),parent(node.parent),children(std::move(node.children)),value(node.value),extent(std::move(node.extent)) {
        node.token=token_type();
        node.parent=nullptr;
        node.value=T0();
    }
    ASTNode(const ASTNode& node): token(node.token),parent(node.parent),children(node.children),value(node.value),extent(node.extent) {
    }
    template <typename TT=T0,typename TL=ASTNodeExtent> ASTNode(token_type t,ASTNode* p=nullptr,const TT& v=TT(),const TL& lines=TL()): token(t), parent(p) {
        extent=lines;
        value=v;
    }
    template <typename TT,typename TL,typename...Args,std::enable_if_t<is_same_type_b<ASTNode*,Args...>&&(!is_same_type_b<TL,ASTNode*>),int> = 0>
    ASTNode(token_type t,ASTNode* p,const TT& v,const TL& lines,ASTNode* c0,Args&&...c): token(t), parent(p) {
    	extent=lines;
        insert(c0,c...);
        value=v;
    }
    template <typename TT,typename...Args,std::enable_if_t<is_same_type_b<ASTNode*,Args...>,int> = 0>
    ASTNode(token_type t,ASTNode* p,const TT& v,ASTNode* c0,Args&&...c): token(t), parent(p) {
        insert(c0,c...);
        value=v;
    }
    ~ASTNode() {
        clear();
    }

    ASTNode& operator=(ASTNode&& node) {
        token=node.token;
        parent=node.parent;
        children=std::move(node.children);
        value=node.value;
        extent=std::move(node.extent);
        node.token=token_type();
        node.parent=nullptr;
        node.value=T0();
        return *this;
    }
    ASTNode& operator=(const ASTNode& node) {
        token=node.token;
        parent=node.parent;
        children=node.children;
        value=node.value;
        extent=node.extent;
        return *this;
    }

    ASTNode*& operator[](size_t i) {
        return children[i];
    }
    const ASTNode*& operator[](size_t i) const {
        return children[i];
    }

    void clear() {
        token=token_type();
        parent=nullptr;
        children.clear();
        value=std::variant_alternative_t<0,variant_type>();
        extent.clear();
    }
    template <typename...Args,std::enable_if_t<is_same_type_b<ASTNode*,Args...>,int> = 0> void insert(ASTNode* c0,Args&&...c) {
        std::initializer_list<ASTNode*> args = {c0,c...};
        for( ASTNode* x : args ) {
            if(x!=nullptr) {
                children.push_back(x);
                children.back()->parent=this;
            }
        }
    }
    template <typename TT,typename...Args,std::enable_if_t<is_same_type_b<ASTNode*,Args...>,int> = 0> void set(token_type t,ASTNode* p,const TT& v,ASTNode* c0,Args&&...c) {
        token=t;
        parent=p;
        children={c0,std::forward(c...)};
        value=v;
    }
    void setLines(int fl,int fc,int ll,int lc) {
        extent.set(fl,fc,ll,lc);
    }
    template <typename TL> void setLines(const TL& begin,const TL& end) {
        extent.set(begin,end);
    }
    template <typename TL,std::enable_if_t<std::is_same<int,TL>::value==false,int> = 0> void setLines(const TL& lines) {
        this->setLines(lines.begin,lines.end);
    }
    template <typename TL,std::enable_if_t<std::is_same<int,TL>::value==true,int> = 0> void setLines(const TL& lines) {
        this->setLines(-1,-1,-1,-1);
    }
    std::ostream& print(std::ostream& ost,std::function<void(std::ostream &,token_type)> tprinter=[](std::ostream& o,token_type t){ o<<t<<": ";},
            std::function<void(std::ostream &,decltype(value))> vprinter=[](std::ostream& o,const decltype(value)& v){ o<<v;},
            std::function<void(std::ostream &,const ASTNodeExtent&)> lprinter=[](std::ostream& o,const ASTNodeExtent& extent){ extent.print(o);},
            std::string sep=" ") const {
        tprinter(ost,token);
        ost<<sep;
        vprinter(ost,value);
        ost<<sep;
        lprinter(ost,extent);
        return ost;
    }
};

template <typename TT,typename T0,typename...T> std::ostream& operator<<(std::ostream& ost,const ASTNode<TT,T0,T...>& node) {
    return node.print(ost);
}
}
#endif
