PROG = run
CPP  = g++
NVCC = nvcc
RM   = /bin/rm
INCLUDE = main.hpp mydef.hpp vector_funcs.hpp malloc.hpp
INCLUDE += input.hpp slicer.hpp output.hpp
VPATH = obj src include results
OBJS = main.o input.o slicer.o output.o
OBJSDIR := ${addprefix ./obj/, ${OBJS}}
CPPFLAGS = -O3 -lm -std=c++11 -I include -I ./

all : ${PROG}

${PROG} : ${OBJS} ${INCLUDE}
	${CPP} -o $@ ${OBJSDIR} ${CPPFLAGS}
#	${NVCC} -o $@ ${OBJSDIR} ${LDFLAGS}

%.o: %.cpp
	${CPP} -c ${CPPFLAGS} $< -o ./obj/$@

#%.o: %.cu
#	${NVCC} -c ${NFLAGS} $< -o ./obj/$@

clean :
	${RM} -f ${PROG} ${OBJSDIR} ./results/*.dat
