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

if [ "$TARCH" = "" ] ; then
  TARCH=cc70
fi

#------------------------------------------------------------------------------
mkdir build
cd build

cmake \
  -DCMAKE_BUILD_TYPE:STRING="$BUILD" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL" \
 \
  -DCMAKE_C_COMPILER:STRING=pgcc \
  -DCMAKE_CXX_COMPILER:STRING=pgc++ \
  -DCMAKE_C_FLAGS:STRING="-O3 -Mlarge_arrays -DNM_VALUE=$NM_VALUE $ALG_OPTIONS -acc -Minfo=accel -ta=multicore" \
\
  -DCMAKE_C_FLAGS_DEBUG:STRING="-g" \
  -DCMAKE_C_FLAGS_RELEASE:STRING="" \
  -DCMAKE_CXX_FLAGS:STRING="-O3 -Mlarge_arrays -DNM_VALUE=$NM_VALUE $ALG_OPTIONS -acc -Minfo=accel -ta=multicore" \
\
  -DCMAKE_CXX_FLAGS_DEBUG:STRING="-g" \
  -DCMAKE_CXX_FLAGS_RELEASE:STRING="" \
 \
  ../
make
make install
#------------------------------------------------------------------------------
