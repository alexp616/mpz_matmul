##### using -fPIC slows things down by a few percent, not a big deal
CC = clang
NVCC = nvcc
CFLAGS = -O3 -fPIC -fomit-frame-pointer -funroll-loops -m64 -pedantic -std=gnu11
NVCCFLAGS = -ccbin $(CC) -arch=sm_89
# debug version
# CFLAGS = -O0 -g -std=gnu11
# NVCCFLAGS = -ccbin gcc -arch=sm_89 -Xcompiler "-g"
LDFLAGS =
INCLUDES = -I/usr/local/include -Iinclude -Iinclude/fft62
LIBS = -L/usr/local/lib /usr/local/lib/libntt-1.0.a -lgmp -lm
INSTALL_ROOT = /usr/local

MPZFFTHEADERS = include/zzcrt.h include/zzmem.h include/zzmisc.h include/mpzfft_moduli.h include/mpnfft.h include/mpnfft_mod.h include/fermat.h include/split.h include/reduce.h include/split_reduce.h include/crt.h include/recompose.h include/crt_recompose.h  include/fft62/arith128.h include/fft62/mod62.h include/fft62/fft62.h include/fft62/cufftwrapper.cuh

MPZFFTOBJECTS = build/zzmisc.o build/moduli.o build/split.o build/reduce.o build/split_reduce.o build/crt.o build/recompose.o build/crt_recompose.o build/mpnfft.o build/fermat.o build/mpnfft_mod.o build/mpzfft.o build/mod62.o build/fft62.o build/zzmem.o build/cufftwrapper.o

RFORESTHEADERS = include/hwmpz.h include/hwmpz_tune.h include/hwmem.h include/rtree.h
RFORESTOBJECTS = build/hwmpz.o build/hwmpz_tune.o build/hwmem.o build/rtree.o build/rforest.o
HEADERS = $(MPZFFTHEADERS) $(RFORESTHEADERS)
OBJECTS = $(MPZFFTOBJECTS) $(RFORESTOBJECTS)
PROGRAMS = test_rforest
TESTS = test_fft

# all: librforest.a $(PROGRAMS) $(TESTS)
all: librforest.a $(PROGRAMS) $(TESTS)

clean:
	rm -f $(shell find . -name '*.o')
	rm -f librforest.a $(PROGRAMS) $(TESTS)

install: all
	cp -v rforest.h $(INSTALL_ROOT)/include
	cp -v librforest.a $(INSTALL_ROOT)/lib

##### rforest library

librforest.a: $(OBJECTS)
	ar -r librforest.a $(OBJECTS)
	ranlib librforest.a
	
##### executables

test_rforest: test_rforest.o librforest.a rforest.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS) librforest.a 

test_fft: build/test_fft.o librforest.a include/fft62/fft62.h include/fft62/cufftwrapper.cuh
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $< $(LIBS) librforest.a

##### tests
test_rforest.o: test_rforest.c include/rforest.h include/hwmem.h include/hwmpz.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

build/test_fft.o: tests/test_fft.c include/fft62/fft62.h include/fft62/cufftwrapper.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

##### hwlpoly modules

build/hwmem.o: src/hwmem.c include/hwmem.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/hwmpz.o: src/hwmpz.c include/hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/hwmpz_tune.o: src/hwmpz_tune.c include/hwmem.h include/hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/rtree.o: src/rtree.c include/rtree.h include/hwmem.h include/hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/rforest.o: src/rforest.c include/rforest.h include/hwmem.h include/hwmpz.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

##### mpzfft C modules

build/zzmisc.o: src/zzmisc.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/moduli.o : src/moduli.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/split.o : src/split.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/reduce.o : src/reduce.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/split_reduce.o : src/split_reduce.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/recompose.o : src/recompose.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/crt.o : src/crt.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/crt_recompose.o : src/crt_recompose.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/mpnfft.o : src/mpnfft.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/fermat.o : src/fermat.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/mpnfft_mod.o : src/mpnfft_mod.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/mpzfft.o : src/mpzfft.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/zzmem.o : src/zzmem.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/mod62.o : src/fft62/mod62.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/fft62.o : src/fft62/fft62.c include/mpzfft.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

build/cufftwrapper.o : src/fft62/cufftwrapper.cu include/fft62/cufftwrapper.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<
##### master header files

mpzfft.h: $(MPZFFTHEADERS)
	touch include/mpzfft.h

rforest.h: $(HEADERS)
	touch include/rforest.h
