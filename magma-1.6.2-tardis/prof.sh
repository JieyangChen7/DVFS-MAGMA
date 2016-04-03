nvprof --profile-from-start off --print-gpu-trace -of profile ./testing/testing_dgeqrf -N 20480,20480
git add profile
git commit -m "updated profile"
git push origin master
