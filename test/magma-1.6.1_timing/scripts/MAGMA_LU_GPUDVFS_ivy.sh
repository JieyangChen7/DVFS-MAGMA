#!/bin/bash

module load cuda/7.0
case=MAGMA_LU_GPUDVFS

#/opt/power-bench/cpuspeed-set.pl --all 1200000
#/home/lchen/cpu_l_ivy.sh
#sleep 9

/opt/power-bench/mclient -H 130.127.48.185 -d /home/lchen/MAGMA/test/magma-1.6.1_timing/results
/opt/power-bench/mclient -H 130.127.48.185 -l ${case}.total.pwr

nvidia-smi -q -d POWER -lms 100 -f /home/lchen/MAGMA/test/magma-1.6.1_timing/results/${case}.gpu.pwr 2>&1 1>/dev/null &
smi_pid=$!
#echo $smi_pid
/usr/local/bin/rapl -c 0,10 -f /home/lchen/MAGMA/test/magma-1.6.1_timing/results/${case}.cpu.pwr 2>&1 1>/dev/null &
rapl_pid=$!
#echo $rapl_pid

#time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgetrf -N 3000,3000 -c
#sudo time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgetrf -N 20000,20000

#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dpotrf -N 5120,5120
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dpotrf -N 10240,10240
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dpotrf -N 15360,15360
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dpotrf -N 20480,20480
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dpotrf -N 25600,25600


#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 5120,5120
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 10240,10240
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 15360,15360
#sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 20480,20480
#sudo time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 10240,10240
#ssh -t -t ivy sudo time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgetrf -N 15360,15360
#sudo time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgetrf -N 20480,20480
##ssh -t -t ivy sudo time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgetrf -N 25600,25600

#sudo time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgeqrf -N 5120,5120
#sudo time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgeqrf -N 10240,10240
#ssh -t -t ivy sudo time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgeqrf -N 15360,15360
#sudo time /home/lchen/MAGMA/test/magma-1.6.1_timing/testing/testing_dgeqrf -N 20480,20480
##ssh -t -t ivy sudo time /home/lchen/MAGMA/magma-1.6.1/testing/testing_dgeqrf -N 25600,25600

/opt/power-bench/mclient -H 130.127.48.185 -e log

kill ${smi_pid}
kill ${rapl_pid}

#sleep 9
#/opt/power-bench/cpuspeed-set.pl --all 2500000
/home/lchen/cpu_h_ivy.sh

scp lchen@ivy2:/home/lchen/MAGMA/test/magma-1.6.1_timing/results/MAGMA_LU_GPUDVFS.total.pwr ./MAGMA_LU_GPUDVFS.total.pwr
