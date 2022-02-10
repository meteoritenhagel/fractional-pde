# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

# CUDA-Toolkit 10* requires g++-8 compilers instead of the most recent g++-9
# solution, according to https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
# > sudo ln -s /usr/bin/g++-8   /usr/local/cuda/bin/g++
# > sudo ln -s /usr/bin/gcc-8   /usr/local/cuda/bin/gcc

ifeq ($(UBUNTU),1)
# on UBUNTU
CUDABIN = /usr/local/cuda-11/bin/
else
# on manjaro
CUDABIN = 
endif

CXX     = $(CUDABIN)nvcc
#CXX     = $(CUDABIN)nvcc -ccbin /usr/bin/g++-10
#F77	= gfortran
LINKER  = ${CXX}


#WARNINGS = -Wall -Weffc++ -Woverloaded-virtual -W -Wfloat-equal -Wshadow \
#           -Wredundant-decls -Winline
#  -Wunreachable-code
WARNINGS =  --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1
#WARNINGS +=  -sp-bound-check -warn-double-usage -warn-lmem-usage -warn-spills -res-usage
#            |--CUDA 7.5   |-- slow !!
# CXXFLAGS += -DNDEBUG ${WARNINGS}

# CUDA
PERF = -use_fast_math -restrict
#DEBUG = -G
DEBUG = -lineinfo

# Ubuntu 16.04 with nvcc V7.5.17  cannot cope with -O1 and higher optimization levels
#CXXFLAGS += -O3  ${PERF} -arch sm_20 --ptxas-options=-v -lineinfo  ${WARNINGS} -I$(SDK_HOME)/inc
#OPT_GPU = -Xptxas --allow-expensive-optimizations
#OPT_GPU = --gpu-architecture compute_62
#CXXFLAGS += -O0  ${PERF} -arch sm_20 --ptxas-options=-v ${OPT_GPU} ${WARNINGS} -I$(SDK_HOME)/inc
#CXXFLAGS += -O0  --std=c++17 --expt-relaxed-constexpr ${PERF}  --ptxas-options=-v ${OPT_GPU} ${WARNINGS} -I$(SDK_HOME)/inc
CXXFLAGS += --std=c++17 --expt-relaxed-constexpr ${PERF}  --ptxas-options=-v ${OPT_GPU} ${WARNINGS} -I$(SDK_HOME)/inc
#-std=c++17

# OpenMP
# https://stackoverflow.com/questions/3211614/using-openmp-in-the-cuda-host-code
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options
# https://gcc.gnu.org/wiki/FloatingPointMath
#CXXFLAGS += --compiler-options=-fopenmp,-O0,-funsafe-math-optimizations
CXXFLAGS += --compiler-options=-fopenmp,-O0,-funsafe-math-optimizations
#CXXFLAGS += -Xcompiler -fopenmp,-O3
LINKFLAG += -lgomp

# BLAS, LAPACK
LINKFLAGS   += -llapack 
ifeq ($(UBUNTU),1)
# on UBUNTU
LINKFLAGS += -lblas
else
# on  manjaro
#LINKFLAGS += -lopenblas -lcblas
LINKFLAGS += -lblas -lcblas
endif
LINKFLAGS   += -lgomp
#LINKFLAGS   += -llapack -lblas
#LINKFLAGS += -lm ${BLAS} -lcudart -lcublas
LINKFLAGS += -lm ${BLAS} -lcudart -lcublas -lcusolver


ifeq ($(MAGMA), TRUE)
# MAGMA
MAGMA_DIR = /usr/local/magma
MAGMA_INCDIR = $(MAGMA_DIR)/include
MAGMA_LIBDIR = $(MAGMA_DIR)/lib

CXXFLAGS += -I$(MAGMA_INCDIR) -DADD_
LINKFLAGS += -L$(MAGMA_LIBDIR) -lmagma
endif

default:	${PROGRAM}

debug: clean
debug: CXXFLAGS += -g
debug: LINKFLAGS += -pg
debug: PERF = 
debug: $(PROGRAM)

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean::
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	-@rm -f *_ *~ *.bak *.log *.out *.tar *.orig
	-@rm -rf html

run: clean ${PROGRAM}
	${OPTIRUN} ./${PROGRAM}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean_all
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

doc:
	doxygen Doxyfile

info:
	inxi -C
	lspci | grep NVIDIA
#	nvidia-smi topo -m
	nvidia-smi
	nvcc -V

#########################################################################
.PRECIOUS: .cu .h
.SUFFIXES: .cu .h .o

.cu.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
#  CXXFLAGS += -g -G
#  LINKFLAGS += -pg
cache: ${PROGRAM}
	${OPTIRUN} $(CUDABIN)nvprof --print-gpu-trace  ./$^ > out_prof.txt
	#${OPTIRUN} nvprof --events l1_global_load_miss,l1_local_load_miss  ./$^ > out_prof.txt

mem: ${PROGRAM}
	${OPTIRUN} $(CUDABIN)cuda-memcheck ./$^

#  Simple run time profiling of your code
#  CXXFLAGS  += -g -G -lineinfo
#  LINKFLAGS += -g -G -lineinfo
#  See also https://docs.nvidia.com/cuda/profiler-users-guide/index.html
prof: ${PROGRAM}
	${OPTIRUN} ./$^
	${OPTIRUN} $(CUDABIN)nvvp ./$^ &

# see also   https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086
prof2: ${PROGRAM}
	$(CUDABIN)nvprof --print-gpu-trace ./$^ 2> prof2.txt

NSYS_OPTIONS = profile --trace=cublas,cuda  --sample=none --cuda-memory-usage=true --cudabacktrace=all --stats=true
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html
prof3: ${PROGRAM}
	$(CUDABIN)nsys $(NSYS_OPTIONS) ./$^ 
	$(CUDABIN)nsys-ui `ls -1tr  report*.qdrep|tail -1`  &
	
prof4: ${PROGRAM}
	$(CUDABIN)nsys-ui ./$^ 
	
	






