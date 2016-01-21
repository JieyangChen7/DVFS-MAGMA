sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda-7.0/bin/nvprof --profile-from-start off --print-gpu-trace --system-profiling on -o profile2 ./testing/testing_dgetrf -N 10240,10240
git add profile2
git commit -m "updated profile2"
git push origin master
