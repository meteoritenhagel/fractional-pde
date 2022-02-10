# Basic Defintions for using GNU-compiler suite sequentially
# requires setting of COMPILER=GCC_

DEPDIR = .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
DEPFILES = $(SOURCES:%.cpp=$(DEPDIR)/%.d)


#CUDA_DIR = /usr/local/cuda-10.2/targets/x86_64-linux
CUDA_DIR = /opt/cuda/targets/x86_64-linux
CUDA_INC = ${CUDA_DIR}/include
CUDA_LIB = ${CUDA_DIR}/lib

CUDAWARNINGS =  --resource-usage -src-in-ptx --restrict --Wreorder --ftemplate-backtrace-limit 1
CUDAPERF = -use_fast_math -restrict
CUDACXXFLAGS += -O3 -arch -std=c++14 --expt-relaxed-constexpr ${CUDAPERF}  --ptxas-options=-v ${OPT_GPU} ${CUDAWARNINGS} -I$(SDK_HOME)/inc

CC	= gcc
CXX     = g++
F77	= gfortran
LINKER  = ${CXX}

WARNINGS = -Wall -pedantic -Wextra -Weffc++ -Woverloaded-virtual  -Wfloat-equal -Wshadow \
           -Wredundant-decls -fmax-errors=1 -Wno-error=redundant-decls
#           -pedantic -Wunreachable-code -Wextra -Winline
#  -Wunreachable-code
#CXXFLAGS += -ffast-math -O3 -march=native -funroll-all-loops ${WARNINGS}
CXXFLAGS += -O3 -std=c++17 ${WARNINGS}
#
# -ftree-vectorizer-verbose=2  -DNDEBUG
# -ftree-vectorizer-verbose=5
# -ftree-vectorize -fdump-tree-vect-blocks=foo.dump  -fdump-tree-pre=stderr

# CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp -fdump-tree-vect-details
# CFLAGS	= -ffast-math -O3 -funroll-loops -DNDEBUG -msse3 -fopenmp -ftree-vectorizer-verbose=2
# #CFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# FFLAGS	= -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
# LFLAGS  = -ffast-math -O3 -DNDEBUG -msse3 -fopenmp
LINKFLAGS += -O3 -flto

# BLAS/LAPACK:
# The <cblas.h> Header should have an 'extern "C"'
#https://gist.github.com/sighingnow/deee806603ec9274fd47
ARCH:= $(shell uname -r|grep arch || echo "ARCH")
ifeq ($(ARCH),ARCH)
LINKFLAGS += -llapack -lopenblas
# -lblas
else
# on  archlinux
LINKFLAGS += -llapack -lopenblas -lcblas
endif

# CUDA, CUBLAS and CUSOLVER support
# CUDA-libs
CXXFLAGS  += -isystem ${CUDA_INC}
LINKFLAGS += -L${CUDA_LIB} -lcudart -lcublas -lcusolver 


OPENMP = -fopenmp
CXXFLAGS += $(OPENMP)
LINKFLAGS += $(OPENMP)

# interprocedural optimization
CXXFLAGS += -flto
LINKFLAGS += -flto

#LINKFLAGS += -DOPENBLAS_NUM_THREADS=2 -DOMP_NUM_THREADS=2

# MAGMA
MAGMA_DIR = /usr/local/magma
MAGMA_INCDIR = $(MAGMA_DIR)/include
MAGMA_LIBDIR = $(MAGMA_DIR)/lib

CXXFLAGS += -isystem $(MAGMA_INCDIR) -DADD_
LINKFLAGS += -L$(MAGMA_LIBDIR) -lmagma

default: ${PROGRAM}

#debug: CXXFLAGS  += -g3
debug: CXXFLAGS  += -g3 -pg
debug: LINKFLAGS  += -pg
debug: ${PROGRAM}

$(DEPDIR):
	mkdir -p $@

# each dependency file as a target, so make won't fail if it doesn't exist yet
#$(DEPFILES):

# include dependencies, wildcard is used to avoid failing on non-existent files
#include $(wildcard $(DEPFILES))

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean:
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	@rm -f *_ *~ *.bak *.log *.out *.aux *.tar *.orig Ns.txt Ms.txt *.csv *.pdf
	@rm -rf html bin obj $(DEPDIR)

run: clean ${PROGRAM}
#	time  ./${PROGRAM}
	./${PROGRAM}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

doc:
	doxygen Doxyfile

#########################################################################

%.o:%.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.o:%.c
	$(CC) -c $(CFLAGS) -o $@ $<
	
%.o:%.cu
	nvcc -c $(CUDAFLAGS) -o $@ $<

%.o:%.f
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################
#    some tools
# Cache behaviour (CXXFLAGS += -g  tracks down to source lines; no -pg in linkflags)
cache: ${PROGRAM}
	valgrind --tool=callgrind --simulate-cache=yes ./$^
#	kcachegrind callgrind.out.<pid> &
	kcachegrind `ls -1tr  callgrind.out.* |tail -1`

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
# no "-pg"  in compile/link options
mem: debug
	valgrind -v --leak-check=yes --tool=memcheck --undef-value-errors=yes --track-origins=yes --log-file=${PROGRAM}.addr.out --show-reachable=yes ./${PROGRAM}

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	./$^
	gprof -b ./$^ > gp.out
#	kprof -f gp.out -p gprof &

#Trace your heap:
#> heaptrack ./main.GCC_
#> heaptrack_gui heaptrack.main.GCC_.<pid>.gz
heap: ${PROGRAM}
	heaptrack ./$^
	heaptrack_gui  `ls -1tr  heaptrack.$^.*.zst |tail -1`
#	heaptrack_gui  `ls -1tr  heaptrack.$^.*.gz |tail -1`

########################################################################
#  get the detailed  status of all optimization flags
info:
	echo "detailed  status of all optimization flags"
	$(CXX) --version
	$(CXX) -Q $(CXXFLAGS) --help=optimizers
	inxi -C
