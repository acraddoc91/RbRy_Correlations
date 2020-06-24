# Makefile template for shared library

CC=g++ # C compiler
CFLAGS=-fPIC -Wall -Wextra -O2 -g -fopenmp # C flags
LDFLAGS=-shared -fopenmp  # linking flags
TARGET_LIB=pythonDLLCPU.so # target lib
SOURCES=pythonDLLCPU.cpp
OBJECTS=$(SOURCES:.cpp=.o)
INC=-I/usr/include/hdf5/serial/ -I/home/sandy/anaconda3/envs/py3/include/python3.6m
LIBS=-lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5 -lsz -lz
LIBLOC=-L/usr/lib/x86_64-linux-gnu/hdf5/serial

all: $(SOURCES) ${TARGET_LIB}

$(TARGET_LIB): $(OBJECTS)
	$(CC) ${LDFLAGS} $(OBJECTS) -o $@ $(LIBLOC) $(LIBS)
	
.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@ $(INC)
