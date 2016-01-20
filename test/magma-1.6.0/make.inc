#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.3.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2012
#//////////////////////////////////////////////////////////////////////////////

#
# GPU_TARGET specifies for which GPU you want to compile MAGMA:
#     "Tesla" (NVIDIA compute capability 1.x cards)
#     "Fermi" (NVIDIA compute capability 2.x cards)
# See http://developer.nvidia.com/cuda-gpus

GPU_TARGET = Kepler

CC        = gcc -DCUBLAS_GFORTRAN
NVCC      = nvcc
FORT      = gfortran -DCUBLAS_GFORTRAN

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O3 -DADD_
F77OPTS   = -O3 -DADD_
FOPTS     = -O3 -DADD_ -x f95-cpp-input
NVOPTS    = -O3 -DADD_ --compiler-options -fno-strict-aliasing -DUNIX
LDOPTS    = -fPIC -Xlinker -zmuldefs

#LIB       = -lf77blas -latlas -lcblas -lg2c -llapack -lcublas -lnvidia-ml -lm
LIB       = -llapack -lf77blas -lgfortran -lcblas -latlas -lcublas -lcudart -lstdc++ -lnvidia-ml -lm

LIB       = -lacml_mp -lcblas -lcublas -lcudart -lstdc++ -lm -lgfortran -lpapi -lnvidia-ml -lpthread

#CUDADIR   = /usr/local/cuda

#LIBDIR    = -L/usr/lib64 -L/usr/lib64/atlas -L/lib64 \
#            -L$(CUDADIR)/lib64 -L/home/lchen/myroot/lib
#INC       = -I$(CUDADIR)/include

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(ACMLDIR)/gfortran64_mp/lib \
            -L$(CBLASDIR)/lib


INC       = -I$(CUDADIR)/include \
            -I$(ACMLDIR)/gfortran64_mp/include \
            -I$(CULA_INC_PATH)
