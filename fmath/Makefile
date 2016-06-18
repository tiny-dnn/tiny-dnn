GCC_VER=$(shell gcc -dumpversion)
ifeq ($(shell expr $(GCC_VER) \>= 4.2),1)
    ADD_OPT+=-mtune=native
endif
ifeq ($(shell expr $(GCC_VER) \>= 4.5),1)
    ADD_OPT+=-fexcess-precision=fast
endif
AVX2=$(shell head -27 /proc/cpuinfo|awk '/avx2/ {print $$1}')
ifeq ($(AVX2),flags)
	HAS_AVX2=-mavx2
endif
# ----------------------------------------------------------------
INC_DIR= -I../src -I../xbyak
CFLAGS += $(INC_DIR) -O3 -D_FILE_OFFSET_BITS=64 -DNDEBUG $(HAS_AVX2) -mfpmath=sse -ffast-math $(ADD_OPT)
CFLAGS_WARN=-Wall -Wextra -Wformat=2 -Wcast-qual -Wcast-align -Wwrite-strings -Wfloat-equal -Wpointer-arith
CFLAGS+=$(CFLAGS_WARN)
# ----------------------------------------------------------------

HEADER= fmath.hpp

TARGET=bench fastexp
all:$(TARGET)

.SUFFIXES: .cpp

bench: bench.o
	$(CXX) -o $@ $<

fastexp: fastexp.o
	$(CXX) -o $@ $<

avx2: avx2.cpp fmath.hpp
	$(CXX) -o $@ $< -Ofast -mavx2 -mtune=native -Iinclude

.cpp.o:
	$(CXX) -c $< -o $@ $(CFLAGS)

.c.o:
	$(CXX) -c $< -o $@ $(CFLAGS)

clean:
	$(RM) *.o $(TARGET)

bench.o: bench.cpp $(HEADER)
fastexp.o: fastexp.cpp $(HEADER)

