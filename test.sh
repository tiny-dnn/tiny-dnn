for error in 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001; do
	echo $error
	g++ -c train.cpp -pthread -std=c++11 -DCNN_USE_GEMMLOWP=true -DCNN_REGISTER_LAYER_DESERIALIZER=true -DADDER_ERROR_RATE=${error} -DMULTI_ERROR_RATE=${error}
	g++ train.o -o train -pthread -std=c++11 -DCNN_USE_GEMMLOWP=true -DCNN_REGISTER_LAYER_DESERIALIZER=true -DADDER_ERROR_RATE=${error} -DMULTI_ERROR_RATE=${error}

	./train examples/mnist/ 2>&1 | tee ${error}.log
done
