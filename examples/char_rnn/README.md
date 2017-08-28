# Building a giter bot with RNNs
In this example, we show how to train a RNN (`gru`, `rnn`, `lstm`) on the chat history of a gitter room and build a bot with it.

## How to test
1. From `build`:
```bash
./examples/example_test_char_rnn --n_layers 3 --hidden_size 256 --backend_type internal --rnn_type gru --seq_len 1 --temperature 0.5
temperature: 1
Hidden size: 256
NLayers: 3
Sequence length: 1
Rnn type: gru
Backend type: Internal

Loading data...
load models...
check 
i did clang accept call can be benced for a gind general be change diff cecmfd .
```

## How to run the bot
0. Requirements: Make sure pyCurl, SWIG, and distutils are installed.
1. Build the tiny_dnn python wrappers:
```bash
cd examples/char_rnn/python
./setup.sh
```

2. Run the gitter server with a room and an API token:
```bash
python3 gitter_server.py --help
usage: gitter_server.py [-h] [--weights_path WEIGHTS_PATH]
                        [--encoding_path ENCODING_PATH] [--depth DEPTH]
                        [--hidden_size HIDDEN_SIZE]
                        [--rnn_type {gru,lstm,rnn}]
                        [--max_output_size MAX_OUTPUT_SIZE]
                        [--softmax_temp SOFTMAX_TEMP]
                        gitter_room gitter_api_token
```
It will start listening to messages with the format @tiny_char_rnn <message>, and using the rnn to answer them.

## How to train
1. cd `examples/char_rnn/python` directory.
2. Download and encode the dataset:
Requirements: python3, this ``gitterpy`` [fork](https://github.com/prlz77/GitterPy)
```bash
python3 prepare_dataset.py --help
usage: prepare_dataset.py [-h] [--gitter_token GITTER_TOKEN]
                          [--chat_room CHAT_ROOM] [--msg_path MSG_PATH]
                          [--encoding_file ENCODING_FILE]
                          [--train_split TRAIN_SPLIT]
                          [--train_output TRAIN_OUTPUT]
                          [--val_output VAL_OUTPUT]
                          [--max_train_size MAX_TRAIN_SIZE] (in bytes)
                          [--max_val_size MAX_VAL_SIZE]
                          seq_len batch_size

```
This will create: `train.raw, train_labels.raw, val.raw, val_labels.raw`, and the encoding dict files.
3. Train the network (cd build dir):
```bash
./examples/example_train_char_rnn --n_layers 3 --hidden_size 256 --backend_type internal --rnn_type gru --seq_len 100
Running with the following parameters:
Learning rate: 0.001
Hidden size: 256
Minibatch size: 32
NLayers: 3
Sequence length: 100
Number of epochs: 20
Rnn type: gru
Dropout rate: 0
Backend type: AVX

Loading data...
load models...
start learning
Train loss: 5.20625
Train loss: 4.09083
Train loss: 3.73759
Train loss: 3.42172
Train loss: 3.12674
Train loss: 2.94797
...
```

4. After every epoch, the validation loss is calculated and the weights are saved if it is the best one.
