rm profile
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda-7.0/bin/nvprof --profile-from-start off --print-gpu-trace --cpu-profiling on -o profile ./testing/testing_dgeqrf -N 20480,20480

git add profile
git commit -m "updated profile"
git push origin master
