#ifndef __AST_HH__
#define __AST_HH__

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <map>
#include <stack>
#include <string>
#include <type_traits>
#include <utility>

#include "astnode.hh"

namespace __ast__ {
enum class astTraversal {
    DFSPreorder,
    DFSPreorderPost
};

template <typename TType,typename T0,typename...TN> class AST {
private:
    std::map<size_t,ASTNode<TType,T0,TN...>> __ast__;
    std::map<ASTNode<TType,T0,TN...>*,size_t> __astLT__;
    size_t __id__=0;
public:
    ASTNode<TType,T0,TN...>* root=nullptr;
    AST(bool insertRoot=true);
    AST(const AST& ast)=delete;
    AST(AST&& ast);
    ~AST();

    AST& operator=(const AST& ast)=delete;
    AST& operator=(AST&& ast);

    std::map<size_t,ASTNode<TType,T0,TN...>>& operator*();
    const std::map<size_t,ASTNode<TType,T0,TN...>>& operator*() const;

    ASTNode<TType,T0,TN...>& operator[](size_t id);
    const ASTNode<TType,T0,TN...>& operator[](size_t id) const;
    size_t operator()(ASTNode<TType,T0,TN...>* node) const;

    std::map<ASTNode<TType,T0,TN...>*,size_t>& LT();
    const std::map<ASTNode<TType,T0,TN...>*,size_t>& LT() const;

    bool valid(size_t id) const;
    bool valid(ASTNode<TType,T0,TN...>* node) const;

    void clear();
    void rebuildLT();
    
    template <typename T=std::variant_alternative_t<0,typename ASTNode<TType,T0,TN...>::variant_type>,typename TL=ASTNodeExtent>
    ASTNode<TType,T0,TN...>& insert(TType token,ASTNode<TType,T0,TN...>* p=nullptr,const T& value=T(),const TL& lines=TL());
    template <typename T=std::variant_alternative_t<0,typename ASTNode<TType,T0,TN...>::variant_type>,typename TL=int,typename...Args,std::enable_if_t<is_same_type_b<ASTNode<TType,T0,TN...>*,Args...>,int> = 0>
    ASTNode<TType,T0,TN...>& insert(TType token,ASTNode<TType,T0,TN...>* p,const T& value,const TL& lines,ASTNode<TType,T0,TN...>* fc,Args...c);
    
    template <typename...Args,std::enable_if_t<is_same_type_b<ASTNode<TType,T0,TN...>*,Args...>,int> = 0> void insert(ASTNode<TType,T0,TN...>* p,ASTNode<TType,T0,TN...>* fc,Args...c);
    
    void erase(size_t id);
    void eraseBranch(size_t id);

    template <astTraversal type=astTraversal::DFSPreorder,typename FT=void,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorder,int> = 0>
    void traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args);
    template <astTraversal type=astTraversal::DFSPreorder,typename FT=void,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorder,int> = 0>
    void traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) const;
    template <astTraversal type=astTraversal::DFSPreorder,typename FT=void,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorderPost,int> = 0>
    void traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args);
    template <astTraversal type=astTraversal::DFSPreorder,typename FT=void,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorderPost,int> = 0>
    void traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) const;
    std::ostream& print(std::ostream& ost,ASTNode<TType,T0,TN...>* node=nullptr,
            std::function<void(const ASTNode<TType,T0,TN...>&,size_t,std::ostream&)> printer=[](const ASTNode<TType,T0,TN...>& node,size_t depth,std::ostream& o) { o<<std::string(depth,' ')<<node<<std::endl;}) const;
    AST deepCopy(ASTNode<TType,T0,TN...>* node=nullptr) const;
};

template <typename TType,typename T0,typename...TN>
AST<TType,T0,TN...>::AST(bool insertRoot) {
	if(insertRoot)
		this->insert(TType());
}
template <typename TType,typename T0,typename...TN>
AST<TType,T0,TN...>::AST(AST &&ast): __ast__(std::move(ast.__ast__)),__astLT__(std::move(ast.__astLT__)), __id__(ast.__id__) {
    ast.__id__=0;
    root=ast.root;
    ast.root=nullptr;
}
template <typename TType,typename T0,typename...TN>
AST<TType,T0,TN...>::~AST() {
    __ast__.clear();
    __astLT__.clear();
    __id__=0;
}

template <typename TType,typename T0,typename...TN>
AST<TType,T0,TN...>& AST<TType,T0,TN...>::operator=(AST &&ast) {
    __ast__=std::move(ast.__ast__);
    __astLT__=std::move(ast.__astLT__);
    root=ast.root;
    ast.root=nullptr;
    __id__=ast.__id__;
    ast.__id__=0;
    return *this;
}

template <typename TType,typename T0,typename...TN>
std::map<size_t,ASTNode<TType,T0,TN...>>& AST<TType,T0,TN...>::operator *() {
    return __ast__;
}
template <typename TType,typename T0,typename...TN>
const std::map<size_t,ASTNode<TType,T0,TN...>>& AST<TType,T0,TN...>::operator *() const {
    return __ast__;
}

template <typename TType,typename T0,typename...TN>
ASTNode<TType,T0,TN...>& AST<TType,T0,TN...>::operator[](size_t id) {
    return __ast__[id];
}
template <typename TType,typename T0,typename...TN>
const ASTNode<TType,T0,TN...>& AST<TType,T0,TN...>::operator[](size_t id) const {
    return __ast__.at(id);
}
template <typename TType,typename T0,typename...TN>
size_t AST<TType,T0,TN...>::operator()(ASTNode<TType,T0,TN...>* node) const {
    return __astLT__.at(node);
}

template <typename TType,typename T0,typename...TN>
std::map<ASTNode<TType,T0,TN...>*,size_t>& AST<TType,T0,TN...>::LT() {
    return __astLT__;
}
template <typename TType,typename T0,typename...TN>
const std::map<ASTNode<TType,T0,TN...>*,size_t>& AST<TType,T0,TN...>::LT() const {
    return __astLT__;
}

template <typename TType,typename T0,typename...TN>
bool AST<TType,T0,TN...>::valid(size_t id) const {
    return __ast__.find(id)!=__ast__.end();
}
template <typename TType,typename T0,typename...TN>
bool AST<TType,T0,TN...>::valid(ASTNode<TType,T0,TN...>* node) const {
    if(node==nullptr)
        return false;
    auto it=__astLT__.find(node);
    if(it!=__astLT__.end())
        if(valid(it->second))
            return it->first==(&__ast__.at(it->second));
    return false;
}

template <typename TType,typename T0,typename...TN>
AST<TType,T0,TN...> AST<TType,T0,TN...>::deepCopy(ASTNode<TType,T0,TN...>* node) const {
	AST copy(false);
    if(node==nullptr)
        node=root;
    if(valid(node)) {
    	std::map<ASTNode<TType,T0,TN...>*,ASTNode<TType,T0,TN...>*> ttable={{node->parent,nullptr}};
    	auto visitor=[&copy,&ttable](const ASTNode<TType,T0,TN...>& n,size_t depth){
    		ttable[const_cast<ASTNode<TType,T0,TN...>*>(&n)]=&copy.insert(n.token,ttable.at(const_cast<ASTNode<TType,T0,TN...>*>(n.parent)),n.value,n.extent);
    	};
    	traverse(visitor,node);
    }
    return copy;
}

template <typename TType,typename T0,typename...TN>
void AST<TType,T0,TN...>::clear() {
    __ast__.clear();
    __astLT__.clear();
    __id__=0;
}
template <typename TType,typename T0,typename...TN>
void AST<TType,T0,TN...>::rebuildLT() {
    __astLT__.clear();
    for(auto it = __ast__.begin(); it != __ast__.end(); ++it)
        __astLT__[&(it->second)]=it->first;
}

template <typename TType,typename T0,typename...TN>
void AST<TType,T0,TN...>::erase(size_t id) {
    assert(valid(id));
    ASTNode<TType,T0,TN...>& self=__ast__[id];
    __astLT__.erase(&self);
    assert(valid(self.parent));
    ASTNode<TType,T0,TN...>& parent=*(self.parent);
    auto it=std::find(parent.children.begin(),parent.children.end(),&self);
    size_t d=std::distance(parent.children.begin(),it);
    parent.children.insert(it+1,self.children.begin(),self.children.end());
    for(ASTNode<TType,T0,TN...>* node : self.children)
    	if(node!=nullptr)
    		node->parent=&parent;
    parent.children.erase(parent.children.begin()+d);
    self.clear();
    __ast__.erase(id);
}
template <typename TType,typename T0,typename...TN>
void AST<TType,T0,TN...>::eraseBranch(size_t id) {
    assert(valid(id));
    __astLT__.erase(&__ast__.at(id));
    assert(valid(__ast__[id].parent));
    ASTNode<TType,T0,TN...>& parent=*(__ast__[id].parent),& self=__ast__[id];
    auto it=std::find(parent.children.begin(),parent.children.end(),&self);
    parent.children.erase(it);
    self.clear();
    __ast__.erase(id);
}

template <typename TType,typename T0,typename...TN> template <typename T,typename TL>
inline ASTNode<TType,T0,TN...>& AST<TType,T0,TN...>::insert(TType token,ASTNode<TType,T0,TN...>* p,const T& value,const TL& lines) {
    if(__ast__.empty()) {
        __ast__[__id__]=ASTNode<TType,T0,TN...>(token,nullptr,value,lines);
        root=&__ast__[__id__];
    }
    else {
        __ast__[__id__]=ASTNode<TType,T0,TN...>(token,p,value,lines);
        if(p!=nullptr)
            p->children.push_back(&__ast__[__id__]);
    }
    ASTNode<TType,T0,TN...>& node=__ast__[__id__];
    __astLT__[&node]=__id__++;
    return node;
}
template <typename TType,typename T0,typename...TN> template <typename T,typename TL,typename...Args,std::enable_if_t<is_same_type_b<ASTNode<TType,T0,TN...>*,Args...>,int>>
inline ASTNode<TType,T0,TN...>& AST<TType,T0,TN...>::insert(TType token,ASTNode<TType,T0,TN...>* p,const T& value,const TL& lines,ASTNode<TType,T0,TN...>* fc,Args...c) {
    if(__ast__.empty()) {
        __ast__[__id__]=ASTNode<TType,T0,TN...>(token,nullptr,value,lines,fc,c...);
        root=&__ast__[__id__];
    }
    else {
        __ast__[__id__]=ASTNode<TType,T0,TN...>(token,p,value,lines,fc,c...);
        if(p!=nullptr)
            p->children.push_back(&__ast__[__id__]);
    }
    ASTNode<TType,T0,TN...>& node=__ast__[__id__];
    for(auto it=node.children.begin();it!=node.children.end();++it) {
        if((*it)!=nullptr) {
            assert(valid(*it));
            (*it)->parent=&node;
        }
    }
    __astLT__[&node]=__id__;
    return __ast__[__id__++];
}
template <typename TType,typename T0,typename...TN> template <typename...Args,std::enable_if_t<is_same_type_b<ASTNode<TType,T0,TN...>*,Args...>,int>>
void AST<TType,T0,TN...>::insert(ASTNode<TType,T0,TN...>* p,ASTNode<TType,T0,TN...>* fc,Args...c) {
    if(valid(p))
        p->insert(fc,c...);
}
template <typename TType,typename T0,typename...TN> template <astTraversal type,typename FT,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorder,int>>
void AST<TType,T0,TN...>::traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) {
    if(node==nullptr)
        node=root;
    if(valid(node)) {
        std::stack<std::pair<ASTNode<TType,T0,TN...>*,size_t>> stack;
        stack.push(std::make_pair(node,0));
        while(!stack.empty()) {
            std::pair<ASTNode<TType,T0,TN...>*,size_t> pnode=stack.top();
            node=pnode.first;
            stack.pop();
            function(*node,pnode.second,args...);
            for(auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
                if((*it)!=nullptr) {
                    assert(valid(*it));
                    stack.push(std::make_pair(*it,pnode.second+1));
                }
            }
        }
    }
}
template <typename TType,typename T0,typename...TN> template <astTraversal type,typename FT,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorder,int>>
void AST<TType,T0,TN...>::traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) const {
    if(node==nullptr)
        node=root;
    if(valid(node)) {
        std::stack<std::pair<ASTNode<TType,T0,TN...>*,size_t>> stack;
        stack.push(std::make_pair(node,0));
        while(!stack.empty()) {
            std::pair<ASTNode<TType,T0,TN...>*,size_t> pnode=stack.top();
            node=pnode.first;
            stack.pop();
            function(*node,pnode.second,args...);
            for(auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
                if((*it)!=nullptr) {
                    assert(valid(*it));
                    stack.push(std::make_pair(*it,pnode.second+1));
                }
            }
        }
    }
}
template <typename TType,typename T0,typename...TN> template <astTraversal type,typename FT,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorderPost,int>>
void AST<TType,T0,TN...>::traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) {
    if(node==nullptr)
        node=root;
    if(valid(node)) {
        std::stack<std::pair<ASTNode<TType,T0,TN...>*,size_t>> stack;
        stack.push(std::make_pair(node,0));
        while(!stack.empty()) {
            std::pair<ASTNode<TType,T0,TN...>*,size_t> pnode=stack.top();
            node=pnode.first;
            stack.pop();
            for(auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
                if((*it)!=nullptr) {
                    assert(valid(*it));
                    stack.push(std::make_pair(*it,pnode.second+1));
                }
            }
            function(*node,pnode.second,args...);
        }
    }
}
template <typename TType,typename T0,typename...TN> template <astTraversal type,typename FT,typename...Args,std::enable_if_t<type==astTraversal::DFSPreorderPost,int>>
void AST<TType,T0,TN...>::traverse(FT& function,ASTNode<TType,T0,TN...>* node,Args&...args) const {
    if(node==nullptr)
        node=root;
    if(valid(node)) {
        std::stack<std::pair<ASTNode<TType,T0,TN...>*,size_t>> stack;
        stack.push(std::make_pair(node,0));
        while(!stack.empty()) {
            std::pair<ASTNode<TType,T0,TN...>*,size_t> pnode=stack.top();
            node=pnode.first;
            stack.pop();
            for(auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
                if((*it)!=nullptr) {
                    assert(valid(*it));
                    stack.push(std::make_pair(*it,pnode.second+1));
                }
            }
            function(*node,pnode.second,args...);
        }
    }
}

template <typename TType,typename T0,typename...TN>
std::ostream& AST<TType,T0,TN...>::print(std::ostream& ost,ASTNode<TType,T0,TN...>* node,std::function<void(const ASTNode<TType,T0,TN...>&,size_t,std::ostream&)> printer) const {
    traverse<astTraversal::DFSPreorder>(printer,node,ost);
    return ost;
}

template <typename TType,typename T0,typename...TN>
std::ostream& operator<<(std::ostream& ost,const AST<TType,T0,TN...>& ast) {
    return ast.print(ost,nullptr);
}

template <typename...> struct ast_tree_type;
template <typename TT,typename T0,typename...TN> struct ast_tree_type<ASTNode<TT,T0,TN...>> {
    typedef AST<TT,T0,TN...> type;
};
template <typename ASTN> using ast_tree_t=typename ast_tree_type<ASTN>::type;
}
#endif
