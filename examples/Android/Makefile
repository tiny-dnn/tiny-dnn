export TINY_CNN_ROOT=`pwd`/../..
.PHONY: all 
CFLAGS=-std=c++11 -Wall

#opencv lib and include 
CFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`

all:cnn_predict.a $(TINY_CNN_ROOT)/lib/libcnn_predict.so exe_predict.a exe_predict

cnn_predict.a:test.cpp
	${CXX} ${CFLAGS} -fPIC -o $@ \
	-I $(TINY_CNN_ROOT) \
        -c $+

exe_predict.a:test.cpp
	$(CXX) $(CFLAGS) -o $@ \
	-I $(TINY_CNN_ROOT) \
	-c $+

exe_predict:exe_predict.a
	@mkdir -p $(TINY_CNN_ROOT)/lib 
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TINY_CNN_ROOT)/lib/libcnn_predict.so: cnn_predict.a
	@mkdir -p $(TINY_CNN_ROOT)/lib
	$(CXX) $(CFLAGS) -shared -o $@ $^ 
	ls -alh $@
clean:
	rm -f *.a $(TINY_CNN_ROOT)/lib/*.so exe_predict
