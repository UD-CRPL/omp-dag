threads?=4
ver?=omp
args?=
BDX?=
BDY?=
BDZ?=
BDO?=
arch?=cc70
dvo?=
dv?=
av?=

all: preprocessor/preprocessor omp serial

omp: wave-dag-omp/bin
wave-dag-omp/bin: preprocessor/preprocessor
	cp -R wavebench-dag wave-dag-omp
	cd wave-dag-omp; ./cmake.sh
	
serial: wave-dag-serial/bin
wave-dag-serial/bin: preprocessor/preprocessor
	cp -R wavebench-dag wave-dag-serial
	cd wave-dag-serial; opts="-s" ./cmake.sh
	
graph: wave-dag-graph/bin
wave-dag-graph/bin: preprocessor/preprocessor
	cp -R wavebench-dag wave-dag-graph
	cd wave-dag-graph;  opts="-p" ./cmake.sh
	
timing: wave-dag-timing/bin 
wave-dag-timing/bin: preprocessor/preprocessor
	cp -R wavebench-dag wave-dag-timing
	cd wave-dag-timing;  opts="-t"  ./cmake.sh

graph-timing: wave-dag-graph-timing/bin
wave-dag-graph-timing/bin: preprocessor/preprocessor
	cp -R wavebench-dag wave-dag-graph-timing
	cd wave-dag-graph-timing;  opts="-t -p" ./cmake.sh

preprocessor/preprocessor :
	make -C preprocessor

acc-gpu: wave-acc-gpu/bin
wave-acc-gpu/bin: 
	cp -R wavebench wave-acc-gpu
	cd wave-acc-gpu;  TARCH=${arch} ./cmake_gpu.sh
	
omp-block: wavebench-omp-block/bin
wavebench-omp-block/bin: 
	cd wavebench-omp-block; ./cmake.sh

omp-naive: wavebench-omp-naive/bin
wavebench-omp-naive/bin: 
	cd wavebench-omp-naive; ./cmake.sh

acc-cpu: wave-acc-cpu/bin
wave-acc-cpu/bin: 
	cp -R wavebench wave-acc-cpu
	cd wave-acc-cpu;  TARCH=${arch} ./cmake_cpu.sh

run: wave-dag-${ver}/bin
	OMP_NUM_THREADS=${threads} \
	OMP_BLOCK_DIMX=${BDX} \
	OMP_BLOCK_DIMY=${BDY} \
	OMP_BLOCK_DIMZ=${BDZ} \
	OMP_BLOCK_DIMO=${BDO} \
	wave-dag-${ver}/bin/wavebench ${args}

run-omp: wavebench-omp-${ver}/bin
	OMP_NUM_THREADS=${threads} \
	OMP_BLOCK_DIMX=${BDX} \
	OMP_BLOCK_DIMY=${BDY} \
	OMP_BLOCK_DIMZ=${BDZ} \
	OMP_BLOCK_DIMO=${BDO} \
	wavebench-omp-${ver}/bin/wavebench ${args}

run-acc: wave-acc-${ver}/bin
	ACC_NUM_CORES=${threads} \
	wave-acc-${ver}/bin/wavebench ${args}

compare-acc: wave-acc-${av}/bin wave-dag-${dv}/bin
	wave-acc-${av}/bin/wavebench ${args} > acc.txt
	OMP_NUM_THREADS=${threads} \
	OMP_BLOCK_DIMX=${BDX} \
	OMP_BLOCK_DIMY=${BDY} \
	OMP_BLOCK_DIMZ=${BDZ} \
	OMP_BLOCK_DIMO=${BDO} \
	wave-dag-${dv}/bin/wavebench ${args} > dag.txt
	diff -U 0 acc.txt dag.txt | grep -v ^@ | wc -l
	
compare-dag: wave-dag-${dvo}/bin wave-dag-${dv}/bin
	OMP_NUM_THREADS=${threads} \
	OMP_BLOCK_DIMX=${BDX} \
	OMP_BLOCK_DIMY=${BDY} \
	OMP_BLOCK_DIMZ=${BDZ} \
	OMP_BLOCK_DIMO=${BDO} \
	wave-dag-${dv}/bin/wavebench ${args} > dag1.txt
	OMP_NUM_THREADS=${threads} \
	OMP_BLOCK_DIMX=${BDX} \
	OMP_BLOCK_DIMY=${BDY} \
	OMP_BLOCK_DIMZ=${BDZ} \
	OMP_BLOCK_DIMO=${BDO} \
	wave-dag-${dvo}/bin/wavebench ${args} > dag2.txt
	diff -U 0 dag1.txt dag2.txt | grep -v ^@ | wc -l

clean:
	rm -Rf wave-dag-omp wave-dag-serial wave-dag-graph wave-dag-timing wave-dag-graph-timing wave-acc-cpu wave-acc-gpu wavebench-omp-naive/build wavebench-omp-naive/bin wavebench-omp-block/build wavebench-omp-block/bin

purge:
	rm -Rf wave-dag-omp wave-dag-serial wave-dag-graph wave-dag-timing wave-dag-graph-timing wave-acc-cpu wave-acc-gpu wavebench-omp-naive/build wavebench-omp-naive/bin wavebench-omp-block/build wavebench-omp-block/bin
	make -C preprocessor purge
