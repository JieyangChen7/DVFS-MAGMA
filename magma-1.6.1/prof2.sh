sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda-7.0/bin/nvprof --profile-from-start off --print-gpu-trace -o profile2 ./testing/testing_dgetrf -N 20480,20480
git add profile2
git commit -m "updated profile2"
git push origin master
