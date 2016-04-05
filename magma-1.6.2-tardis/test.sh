git pull origin master
make -j 32 -s
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./testing/testing_dgeqrf -N 20480,20480
