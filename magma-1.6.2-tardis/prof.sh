
#sudo cannot access profile, so we need to modify its permission
rm profile
touch profile
chmod 777 profile
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH env PATH=$PATH nvprof --profile-from-start off --print-gpu-trace -o profile ./testing/testing_dgeqrf -N 20480,20480

git add profile
git commit -m "updated profile"
git push origin master
