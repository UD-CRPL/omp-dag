%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cstdlib>
# include <cstring>
# include <string>
# include "driver.hh"
# include "parser.hh"
%}

%option noyywrap nounput noinput batch debug

blank [ \t\r]

%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}
%%
%{
  // A handy shortcut to the location held by the Driver.
  yy::location& loc = drv.location;
  // Code run each time yylex is called.
  loc.step ();
%}
{blank}+        loc.step ();
\n+             loc.lines (yyleng); loc.step ();

"("             return yy::parser::make_LPAREN (loc);
")"             return yy::parser::make_RPAREN (loc);
"{"             return yy::parser::make_LBRACE (loc);
"}"             return yy::parser::make_RBRACE (loc);
^"#pragma"      return yy::parser::make_PRAGMA (loc);
"omp"           return yy::parser::make_OMP(loc);
"dag"           return yy::parser::make_DAG (loc);
"num_threads"   return yy::parser::make_NUM_THREADS (loc);
"graph"         return yy::parser::make_GRAPH (loc);
"task"          return yy::parser::make_TASK (loc);
"depend"        return yy::parser::make_DEPEND (loc);
"coarsening"    return yy::parser::make_COARSENING (loc);
"DEFAULT"       return yy::parser::make_DEFAULT (loc);
"BFS"           return yy::parser::make_BFS (loc);
"DFS"           return yy::parser::make_DFS (loc);
"METIS"         return yy::parser::make_METIS (loc);
"TREE"          return yy::parser::make_TREE (loc);
"BLOCK"         return yy::parser::make_BLOCK (loc);
"CUSTOM"        return yy::parser::make_CUSTOM (loc);
"->"            return yy::parser::make_ARROW (loc);
","             return yy::parser::make_COMMA (loc);

.               return yy::parser::make_ANY (yytext, loc);

<<EOF>>    return yy::parser::make_END (loc);
%%
void
Driver::scanBegin () {
    yy_flex_debug = trace_scanning;
    if (file.empty () || file == "-")
        yyin = stdin;
    else if (!(yyin = fopen (file.c_str (), "r"))) {
        std::cerr << "cannot open " << file << ": " << strerror(errno) << '\n';
        exit (EXIT_FAILURE);
    }
}

void
Driver::scanEnd () {
    fclose (yyin);
}
