quantized:
	g++ -c quantized.cpp -pthread -std=c++11 -D CNN_USE_GEMMLOWP -D CNN_REGISTER_LAYER_DESERIALIZER
	g++ quantized.o -o quantized -pthread -std=c++11 -D CNN_USE_GEMMLOWP -D CNN_REGISTER_LAYER_DESERIALIZER

train:
	g++ -c train.cpp -pthread -std=c++11 -DCNN_USE_GEMMLOWP=true -DCNN_REGISTER_LAYER_DESERIALIZER=true -DADDER_ERROR_RATE=0.00001 -DMULTI_ERROR_RATE=0.00001
	g++ train.o -o train -pthread -std=c++11 -DCNN_USE_GEMMLOWP=true -DCNN_REGISTER_LAYER_DESERIALIZER=true -DADDER_ERROR_RATE=0.00001 -DMULTI_ERROR_RATE=0.00001

test:
	g++ -c test.cpp -pthread -std=c++11 -D CNN_USE_GEMMLOWP -D CNN_REGISTER_LAYER_DESERIALIZER
	g++ test.o -o test -pthread -std=c++11 -D CNN_USE_GEMMLOWP -D CNN_REGISTER_LAYER_DESERIALIZER

clean:
	rm test.o train.o quantized.o test train quantized

