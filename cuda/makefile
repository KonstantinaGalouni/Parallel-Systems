CUDA_INSTALL_PATH = /usr/local/cuda
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I./common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = cuda_mpi_heat2D cuda_mpi_heat2D.o

cuda_mpi_heat2D: cuda_mpi_heat2D.o
	${CC} ${LFLAGS} -o $@ $^

cuda_mpi_heat2D.o: cuda_mpi_heat2D.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@	-arch=sm_20	
 
clean:
	\rm -f $(CLEAN_FILES)	
	\rm -f ../release/cuda_mpi_heat2D

rebuild: clean cuda_mpi_heat2D
