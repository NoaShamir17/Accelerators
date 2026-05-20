DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g -G
else
CFLAGS=-O3 -lineinfo
endif

CFLAGS+=-Xcompiler=-Wall -maxrregcount=64 -arch=sm_75
CFLAGS+=`pkg-config opencv --cflags --libs`

FILES=ex1 image

OBJS=ex1.o main.o image.o ex1-cpu.o

all: $(FILES)

ex1: ex1.o main.o ex1-cpu.o 
	nvcc --link $(CFLAGS) -L. -lutils ex1.o main.o ex1-cpu.o -o $@

image: ex1-cpu.o image.o
	nvcc --link $(CFLAGS) $^ -o $@

ex1.o: ex1.cu ex1.h
main.o: main.cu ex1.h
image.o: image.cu ex1.h

%.o: %.cu
	nvcc --compile -dc $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
