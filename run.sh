#!/bin/bash

__CORES__=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}' )
__THREADS__=$(grep -c ^processor /proc/cpuinfo)

#tnum="${tnum:-$__CORES__}"
tnum="${tnum:-$__THREADS__}"

function is_power_of_two () {
    declare -i n=$1
    (( n > 0 && (n & (n - 1)) == 0 ))
}


function run_cores() {
i=2
if is_power_of_two "$tnum" ; then
	while [ "$i" -le  "$tnum" ]; do
		echo "__cores__=$i"
		"$@" $i
		i=$(( i * 2 ))
	done
else
	while [ "$i" -le  "$tnum" ]; do
		echo "__cores__=$i"
		"$@" $i
		i=$(( i +2 ))
	done
fi
}

function run_dag() {
j=$1
e=$2
ver=$3
opts=$4
bdx=$5
bdy=$6
bdz=$7
tn=$8
while [ "$j" -le "$e"  ]; do
	echo "__input_size__=$j"
	make run ver="$ver" args="--ncell_x $j --ncell_y $j --ncell_z $j $opts" "$bdx" "$bdy" "$bdz" threads="$tn"
	j=$(( j * 2 ))
done
}
function run_acc() {
j=$1
e=$2
ver=$3
opts=$4
tn=$5
while [ "$j" -le "$e"  ]; do
	echo "__input_size__=$j"
	make run-acc ver="$ver" args="--ncell_x $j --ncell_y $j --ncell_z $j $opts" threads="$tn"
	j=$(( j * 2 ))
done
}
date
run_cores run_dag 8 64 "omp" "--alg 0 --print 0" "BDX=2" "BDY=2" "BDZ=2" >>  rt-dag.txt 2>&1
run_cores run_dag 4096 32768 "omp" "--alg 1 --print 0 --ncomp 1" "BDX=256" "BDY=256" "BDZ=256" >>  sw-dag.txt 2>&1

run_cores run_acc 8 64 "gpu" "--alg 0 --print 0" >>  rt-gacc.txt 2>&1
run_cores run_acc 4096 32768 "gpu" "--alg 1 --print 0 --ncomp 1" >>  sw-gacc.txt 2>&1

run_cores run_acc 8 64 "cpu" "--alg 0 --print 0" >>  rt-cacc.txt 2>&1
run_cores run_acc 4096 32768 "cpu" "--alg 1 --print 0 --ncomp 1" >>  sw-cacc.txt 2>&1

run_dag 8 64 "serial" "--alg 0 --print 0" "BDX=2" "BDY=2" "BDZ=2" 1 >>  rt-sdag.txt 2>&1
run_dag 4096 32768 "serial" "--alg 1 --print 0 --ncomp 1" "BDX=128" "BDY=128" "BDZ=128" 1 >>  sw-sdag.txt 2>&1
date
