#
# use GNU-Compiler tools
#COMPILER=GCC_CPU_ONLY_

COMPILER=NVCC_

MAGMA=FALSE

# alternatively from the shell
# export COMPILER=GCC_
# or, alternatively from the shell
# make COMPILER=GCC_

# use Intel compilers
#COMPILER=ICC_

# use PGI compilers
# COMPILER=PGI_

SOURCES = src/main.cpp src/processingunit/timer.cpp

ifneq ($(COMPILER), ONEAPI_CPU_ONLY_)
ifneq ($(COMPILER), GCC_CPU_ONLY_)
  SOURCES += src/processingunit/gpu_handle.cpp src/devicedata/initializememory.cu src/gpukernels.cu
endif
endif

ifeq ($(MAGMA), TRUE)
SOURCES += src/processingunit/gpu_magma_queue.cpp
endif

TEMP_OBJECTS = $(SOURCES:.cu=.o)
OBJECTS = $(TEMP_OBJECTS:.cpp=.o)

PROGRAM	= main.${COMPILER}

# uncomment the next to lines for debugging and detailed performance analysis
#CXXFLAGS += -g -G -pg
#LINKFLAGS += -g -G -pg
CXXFLAGS += -g -pg
#  /usr/include/lapack.h:19   //#define LAPACK_FORTRAN_STRLEN_END
#  deactivated by GH
# -ULAPACK_FORTRAN_STRLEN_END
#LINKFLAGS += -g -pg
# -llapack -lblas
# do not use -pg with PGI compilers

ifndef COMPILER
  COMPILER=GCC_
endif

include ./makefiles/${COMPILER}default.mk
#-llapacke
