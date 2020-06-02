#!/bin/bash -l
#------------------------------------------------------------------------------

# CLEANUP
rm -rf CMakeCache.txt
rm -rf CMakeFiles
rm -rf bin/

#opts=""

if [ "$INSTALL" = "" ] ; then
  INSTALL=../
fi

if [ "$BUILD" = "" ] ; then
  BUILD=Release
fi

if [ "$NM_VALUE" = "" ] ; then
  NM_VALUE=4
fi

#------------------------------------------------------------------------------

bash -c "cd src/algorithms; ../../../preprocessor/preprocessor local_sequence_alignment.cpp -o local_sequence_alignment.cpp -c \"-std=c++17 -I../main\" ${opts}"
bash -c "cd src/algorithms; ../../../preprocessor/preprocessor radiation_transport.cpp -o radiation_transport.cpp -c \"-std=c++17 -I../main\" ${opts}"

mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE:STRING="$BUILD" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL" \
 \
  -DCMAKE_C_COMPILER:STRING=gcc-9 \
  -DCMAKE_CXX_COMPILER:STRING=g++-9 \
  -DCMAKE_C_FLAGS:STRING="-O3 -D_OPENMP=201811 -fopenmp -DNM_VALUE=$NM_VALUE $ALG_OPTIONS" \
\
  -DCMAKE_C_FLAGS_DEBUG:STRING="-g" \
  -DCMAKE_C_FLAGS_RELEASE:STRING="-w" \
  -DCMAKE_CXX_FLAGS:STRING="-O3 -D_OPENMP=201811 -fopenmp -DNM_VALUE=$NM_VALUE $ALG_OPTIONS" \
\
  -DCMAKE_CXX_FLAGS_DEBUG:STRING="-g" \
  -DCMAKE_CXX_FLAGS_RELEASE:STRING="-w" \
 \
  ../
make
make install
#------------------------------------------------------------------------------
