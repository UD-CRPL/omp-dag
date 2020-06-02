%skeleton "lalr1.cc" /* -*- C++ -*- */
//%require "3.5.1"
%defines

%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires {
  #include <iostream>
  #include <map>
  #include <string>
  enum class PRAGMA_TOKENS_E {
    ROOT=-1,
    PRAGMA=1, GRAPH=2, COARSENING=4, NUM_THREADS=8, TASK=16, DEPEND=32,
    DEPENDENCY=64, SIMPLE_DEPENDENCY, CONDITIONAL_DEPENDENCY, SIMPLE_CONDITIONAL_DEPENDENCY, DEPENDENCY_LIST, 
    COARSENING_OPTS, DEFAULT, BFS, DFS, METIS, TREE, BLOCK, CUSTOM,
    SYMBOL
  };
  typedef PRAGMA_TOKENS_E pragma_tokens;
  using ptok_t=pragma_tokens;
  std::ostream& operator<<(std::ostream& ost,pragma_tokens V);
  pragma_tokens operator|(pragma_tokens a, pragma_tokens b);
  pragma_tokens operator&(pragma_tokens a, pragma_tokens b);
  #include "ast/ast.hh"
  using pragma_ast_node_t=__ast__::ASTNode<pragma_tokens,int,std::string,pragma_tokens>;
  using pragma_ast_t=__ast__::ast_tree_t<pragma_ast_node_t>;
  class Driver;
}
%code {
  const std::map<pragma_tokens,std::string> tokens_names = { {pragma_tokens::ROOT,"ROOT"},
  {pragma_tokens::PRAGMA,"pragma omp dag"}, {pragma_tokens::GRAPH,"graph"}, {pragma_tokens::COARSENING,"coarsening"}, {pragma_tokens::NUM_THREADS,"num_threads"}, {pragma_tokens::TASK,"task"}, {pragma_tokens::DEPEND,"depend"},
  {pragma_tokens::DEPENDENCY,"dependency"}, {pragma_tokens::SIMPLE_DEPENDENCY,"simple dependency"}, {pragma_tokens::CONDITIONAL_DEPENDENCY,"conditional dependency"}, {pragma_tokens::SIMPLE_CONDITIONAL_DEPENDENCY,"simple conditional dependency"}, {pragma_tokens::DEPENDENCY_LIST,"dependency list"}, 
  {pragma_tokens::COARSENING_OPTS,"coarsening options"}, {pragma_tokens::DEFAULT,"default"}, {pragma_tokens::BFS,"bfs"}, {pragma_tokens::DFS,"dfs"}, {pragma_tokens::METIS,"metis"}, {pragma_tokens::TREE,"tree"}, {pragma_tokens::BLOCK,"block"}, {pragma_tokens::CUSTOM,"custom"}, 
  {pragma_tokens::SYMBOL,"symbol"}
  };
  std::ostream& operator<<(std::ostream& ost,pragma_tokens V) {
    if(tokens_names.find(V)==tokens_names.end())
        ost<<static_cast<int>(V);
    else
        ost<<tokens_names.at(V);
    return ost;
  }
  pragma_tokens operator|(pragma_tokens a, pragma_tokens b) {
    return static_cast<pragma_tokens>(static_cast<int>(a)|static_cast<int>(b));
  }
  pragma_tokens operator&(pragma_tokens a, pragma_tokens b) {
    return static_cast<pragma_tokens>(static_cast<int>(a)&static_cast<int>(b));
  }
}
// The parsing context.
%param { Driver& drv }

%locations

%define parse.trace
%define parse.error verbose

%code {
#include "driver.hh"
}

%define api.token.prefix {TOK_}
%token
  END  0        "end of file"
  LPAREN        "("
  RPAREN        ")"
  LBRACE        "{"
  RBRACE        "}"
  ARROW         "->"
  COMMA         ","
  PRAGMA
  OMP
  NUM_THREADS
  GRAPH
  DAG
  TASK
  DEPEND
  COARSENING
  BFS
  DFS
  METIS
  DEFAULT
  TREE
  BLOCK
  CUSTOM
;

%token <std::string> ANY
%nterm <pragma_ast_node_t*> unit
%nterm <pragma_ast_node_t*> body
%nterm <pragma_ast_node_t*> body_helper
%nterm <pragma_ast_node_t*> discarded
%nterm <pragma_ast_node_t*> pragma
%nterm <pragma_ast_node_t*> dependency_list
%nterm <pragma_ast_node_t*> dependency
%nterm <pragma_ast_node_t*> coarsening_method
%nterm <pragma_ast_node_t*> coarsening_opts
%nterm <std::string> expression
%nterm <std::string> symbol
%nterm <std::string> basic_symbol
%nterm <std::string> function_symbol
%nterm <std::string> symbol_list

%printer { yyo << $$; } <*>;

%precedence PRAGMA
%precedence OMP
%precedence DAG
%precedence GRAPH
%precedence COARSENING
%precedence NUM_THREADS
%precedence TASK
%precedence DEPEND

%%
%start unit;
unit: body { };

body: %empty {}
    |   body body_helper    {}
    ;

body_helper: discarded       {}
    |        pragma   {}
    ;
    
pragma: PRAGMA                                                                                              {}
    |   PRAGMA OMP                                                                                          {}
    |   PRAGMA OMP DAG                                                                                      { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA,@$);
        }
    |   PRAGMA OMP DAG NUM_THREADS "(" symbol ")"                                                           { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::NUM_THREADS,@$,&drv.ast.insert(ptok_t::NUM_THREADS,nullptr,$6,@6)); 
        }
    |   PRAGMA OMP DAG COARSENING "(" coarsening_method ")"                                                 { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::COARSENING,@$,&drv.ast.insert(ptok_t::COARSENING,nullptr,0,@4,$6)); 
        }
    |   PRAGMA OMP DAG COARSENING "(" coarsening_method ")" NUM_THREADS "(" symbol ")"                      { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::COARSENING|ptok_t::NUM_THREADS,@$,&drv.ast.insert(ptok_t::COARSENING,nullptr,0,@4,$6),&drv.ast.insert(ptok_t::NUM_THREADS,nullptr,$10,@10));
        }
    |   PRAGMA OMP DAG GRAPH "(" symbol ")"                                                                 { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::GRAPH,@$,&drv.ast.insert(ptok_t::GRAPH,nullptr,$6,@6)); 
        }
    |   PRAGMA OMP DAG GRAPH "(" symbol ")" NUM_THREADS "(" symbol ")"                                      { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::GRAPH|ptok_t::NUM_THREADS,@$,&drv.ast.insert(ptok_t::GRAPH,nullptr,$6,@6),&drv.ast.insert(ptok_t::NUM_THREADS,nullptr,$10,@10)); 
        }
    |   PRAGMA OMP DAG GRAPH "(" symbol ")" COARSENING "(" coarsening_method ")"                            {
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::GRAPH|ptok_t::COARSENING,@$,&drv.ast.insert(ptok_t::GRAPH,nullptr,$6,@6),&drv.ast.insert(ptok_t::COARSENING,nullptr,0,@8,$10)); 
        }
    |   PRAGMA OMP DAG GRAPH "(" symbol ")" COARSENING "(" coarsening_method ")" NUM_THREADS "(" symbol ")" { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::GRAPH|ptok_t::COARSENING|ptok_t::NUM_THREADS,@$,&drv.ast.insert(ptok_t::GRAPH,nullptr,$6,@6),
                                 &drv.ast.insert(ptok_t::COARSENING,nullptr,0,@8,$10),&drv.ast.insert(ptok_t::NUM_THREADS,nullptr,$14,@14));
        }
    |   PRAGMA OMP DAG TASK                                                                                 { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::TASK,@$,&drv.ast.insert(ptok_t::TASK,nullptr,0,@4)); 
        }
    |   PRAGMA OMP DAG TASK DEPEND "(" dependency_list ")"                                                  { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::TASK|ptok_t::DEPEND,@$,&drv.ast.insert(ptok_t::TASK,nullptr,0,@4,&drv.ast.insert(ptok_t::DEPEND,nullptr,0,@5,$7))); 
        }
    |   PRAGMA OMP DAG DEPEND "(" dependency_list ")"                                                       { 
            $$ = &drv.ast.insert(ptok_t::PRAGMA,drv.ast.root,ptok_t::PRAGMA|ptok_t::DEPEND,@$,&drv.ast.insert(ptok_t::DEPEND,nullptr,0,@4,$6)); 
        }
    ;

coarsening_method: DEFAULT                   { $$ = &drv.ast.insert(ptok_t::DEFAULT,nullptr,0,@$); }
    |              METIS                     { $$ = &drv.ast.insert(ptok_t::METIS,nullptr,0,@$); }
    |              BFS                       { $$ = &drv.ast.insert(ptok_t::BFS,nullptr,0,@$); }
    |              DFS                       { $$ = &drv.ast.insert(ptok_t::DFS,nullptr,0,@$); }
    |              BLOCK                     { $$ = &drv.ast.insert(ptok_t::BLOCK,nullptr,0,@$); }
    |              TREE                      { $$ = &drv.ast.insert(ptok_t::TREE,nullptr,0,@$); }
    |              METIS "," coarsening_opts { $$ = &drv.ast.insert(ptok_t::METIS,nullptr,0,@3,$3); }
    |              BFS "," coarsening_opts   { $$ = &drv.ast.insert(ptok_t::BFS,nullptr,0,@3,$3); }
    |              DFS "," coarsening_opts   { $$ = &drv.ast.insert(ptok_t::DFS,nullptr,0,@3,$3); }
    |              BLOCK "," coarsening_opts { $$ = &drv.ast.insert(ptok_t::BLOCK,nullptr,0,@3,$3); }
    |              TREE "," coarsening_opts  { $$ = &drv.ast.insert(ptok_t::TREE,nullptr,0,@3,$3); }
    |              CUSTOM "," symbol         { $$ = &drv.ast.insert(ptok_t::CUSTOM,nullptr,$3,@3); }
    ;
    
coarsening_opts: expression                     { $$ = &drv.ast.insert(ptok_t::COARSENING_OPTS,nullptr,0,@$,&drv.ast.insert(ptok_t::SYMBOL,nullptr,$1,@1)); }
    |            coarsening_opts "," expression { $$ = $1; $1->insert(&drv.ast.insert(ptok_t::SYMBOL,nullptr,$3,@3)); }
    ;
    
dependency_list: dependency                     { $$ = &drv.ast.insert(ptok_t::DEPENDENCY_LIST,nullptr,0,@$,$1); }
    |            dependency_list "," dependency { $$ = $1; $1->insert($3); }
    ;
    
dependency: "{" expression "," expression "}"                 { 
                    $$ = &drv.ast.insert(ptok_t::DEPENDENCY,nullptr,ptok_t::SIMPLE_CONDITIONAL_DEPENDENCY,@$,&drv.ast.insert(ptok_t::SYMBOL,nullptr,$2,@2),&drv.ast.insert(ptok_t::SYMBOL,nullptr,$4,@4));
                }
    |       "{" expression "->" expression "," expression "}" { 
                    $$ = &drv.ast.insert(ptok_t::DEPENDENCY,nullptr,ptok_t::CONDITIONAL_DEPENDENCY,@$,&drv.ast.insert(ptok_t::SYMBOL,nullptr,$2,@2),&drv.ast.insert(ptok_t::SYMBOL,nullptr,$4,@4),&drv.ast.insert(ptok_t::SYMBOL,nullptr,$6,@6)); 
                }
    |       expression "->" expression                        { 
                    $$ = &drv.ast.insert(ptok_t::DEPENDENCY,nullptr,ptok_t::DEPENDENCY,@$,&drv.ast.insert(ptok_t::SYMBOL,nullptr,$1,@1),&drv.ast.insert(ptok_t::SYMBOL,nullptr,$3,@3)); 
                }
    |       expression                                        {
                    $$ = &drv.ast.insert(ptok_t::DEPENDENCY,nullptr,ptok_t::SIMPLE_DEPENDENCY,@$,&drv.ast.insert(ptok_t::SYMBOL,nullptr,$1,@1));
                }
    ;
    
expression: symbol        { $$ = $1; }
    |       "(" expression ")" { $$ = "("+$2+")"; }
    |       "(" expression ")" symbol { $$ = "("+$2+")"+$4; }
    ;

symbol: basic_symbol           { $$=$1; }
    |   function_symbol        { $$=$1; }
    ;

function_symbol: basic_symbol "(" symbol_list ")" { $$=$1+"("+$3+")"; }
    ;

symbol_list: expression { $$=$1; }
    |        symbol_list "," expression { $$=$1+","+$3; }
    ;

basic_symbol: ANY                          { $$=$1; }
    |         basic_symbol ANY             { $$=$1+$2; }
    ;

discarded: OMP            {}
    |      DAG            {}
    |      ANY            {}
    |      LPAREN         {}
    |      RPAREN         {}
    |      LBRACE         {}
    |      RBRACE         {}
    |      ARROW          {}
    |      COMMA          {}
    |      GRAPH          {}
    |      NUM_THREADS    {}
    |      TASK           {}
    |      DEPEND         {}
    |      COARSENING     {}
    |      BFS            {}
    |      DFS            {}
    |      METIS          {}
    |      DEFAULT        {}
    |      TREE           {}
    |      BLOCK          {}
    |      CUSTOM         {}
    ;

%%
void
yy::parser::error (const location_type& l, const std::string& m) {
  std::cerr << l << ": " << m << '\n';
}
