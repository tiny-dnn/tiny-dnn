cd tiny_dnn
swig -c++ -python tiny_char_rnn.i
python3 setup.py build_ext --inplace
