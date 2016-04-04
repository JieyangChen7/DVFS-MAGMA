rm profile
nvprof --profile-from-start off --print-gpu-trace --cpu-profiling on -o profile ./testing/testing_dgeqrf --nthread 1 -N 20480,20480
git add profile
git commit -m "updated profile"
git push origin master
