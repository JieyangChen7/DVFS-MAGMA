sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda-7.0/bin/nvprof --profile-from-start off --print-gpu-trace -o profile ./testing/testing_dgetrf -N 10240,10240
git add profile
git commit -m "updated profile"
git push origin master
