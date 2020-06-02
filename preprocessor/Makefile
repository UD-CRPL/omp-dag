BASE = preprocessor
BISON = bison
CXX = g++-9
FLEX = flex
CXXFLAGS ?= -std=c++17

all: $(BASE)

$(BASE):
	mkdir -v build
	$(MAKE) -C parser-lib CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) BISON=$(BISON) FLEX=$(FLEX) static
	mv -v parser-lib/include build/include
	mv -v parser-lib/lib build/lib
	$(MAKE) -C parser-lib purge
	cp -Rv preprocessor-src/* build/
	$(MAKE) -C build CXX=$(CXX) CXXFLAGS=$(CXXFLAGS)
	mv -v build/preprocessor preprocessor
	$(MAKE) clean

clean:
	rm -Rf build
	$(MAKE) -C parser-lib purge
	
purge:
	rm -Rf build preprocessor
	$(MAKE) -C parser-lib purge
