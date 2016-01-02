#!/bin/bash

##./file_clean $1.cpu.pwr
grep "Power Draw" < $1.gpu.pwr | sed 's/[^0-9.]*//g' > gpupower1.out
##sleep 1
####awk '{ sum += $1 } END { print sum }' gpupower1.out
##sed ':a;N;s/\n/+/;ta' gpupower1.out|bc
./energy_calc $1 0 0
