# This Makefile is designed to be simple and readable.  It does not
# aim at portability.  It requires GNU Make.

BASE = pragma-parser
BISON = bison
CXX = g++-9
FLEX = flex
XSLTPROC = xsltproc
CXXFLAGS ?= -std=c++17

all: static

%.cc %.hh %.xml %.gv: %.yy
	$(BISON) $(BISONFLAGS) --xml --graph=$*.gv -o $*.cc $<

%.cc: %.ll
	$(FLEX) $(FLEXFLAGS) -o$@ $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o$@ $<

shared: CXXFLAGS = -std=c++17 -fPIC 
shared: $(BASE).o parser.o driver.o scanner.o
	$(CXX) -shared astextent.o parser.o driver.o scanner.o $(BASE).o -o libpragma-parser.so
	mkdir -v lib
	mv -v libpragma-parser.so lib/
	mkdir -v include
	cp -v pragma-parser.hh include/
	cp -v parser.hh include/
	cp -v location.hh include/
	cp -v driver.hh include/
	ln -sv ${PWD}/ast include/
	$(MAKE) clean

static: astextent.o parser.o driver.o scanner.o $(BASE).o
	ar rcs libpragma-parser.a astextent.o parser.o driver.o scanner.o $(BASE).o
	mkdir -v lib
	mv -v libpragma-parser.a lib/
	mkdir -v include
	cp -v pragma-parser.hh include/
	cp -v parser.hh include/
	cp -v location.hh include/
	cp -v driver.hh include/
	cp -Rv ast include/
	$(MAKE) clean
	
astextent.o: ast/astextent.cc
	$(CXX) $(CXXFLAGS) -c -o$@ $<
$(BASE).o: parser.hh
parser.o: parser.hh
scanner.o: parser.hh

html: parser.html
%.html: %.xml
	$(XSLTPROC) $(XSLTPROCFLAGS) -o $@ $$($(BISON) --print-datadir)/xslt/xml2xhtml.xsl $<

CLEANFILES =										\
  $(BASE) *.o										\
  parser.hh parser.cc parser.output parser.xml parser.html parser.gv location.hh	\
  scanner.cc position.hh stack.hh

clean:
	rm -f $(CLEANFILES)
purge:
	rm -f $(CLEANFILES)
	rm -rf ./lib
	rm -rf ./include
