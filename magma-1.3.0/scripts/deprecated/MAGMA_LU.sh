#!/bin/sh

case=MAGMA_LU_2601000

/usr/local/bin/mclient -H 10.1.255.101 -d /home/lchen/MAGMA/magma-1.3.0/results
/usr/local/bin/mclient -H 10.1.255.101 -l ${case}.total.pwr

nvidia-smi -q -d POWER -lms 100 -f /home/lchen/MAGMA/magma-1.3.0/results/${case}.gpu.pwr 2>&1 1>/dev/null &
smi_pid=$!
#echo $smi_pid
/apps/power-bench/rapl -c 0,8 -f /home/lchen/MAGMA/magma-1.3.0/results/${case}.cpu.pwr 2>&1 1>/dev/null &
rapl_pid=$!
#echo $rapl_pid

#/apps/power-bench/setcpuspeed sandy 1200000
#echo 2601000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
/apps/power-bench/setcpuspeed sandy 2601000

#time /home/lchen/MAGMA/magma-1.3.0/testing/testing_dgetrf -N 3000,3000 -c
time /home/lchen/MAGMA/magma-1.3.0/testing/testing_dgetrf -N 20000,20000
##sleep 9

/usr/local/bin/mclient -H 10.1.255.101 -e log

kill ${smi_pid}
kill ${rapl_pid}

/apps/power-bench/setcpuspeed sandy 2600000
