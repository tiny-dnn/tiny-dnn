# Recurrent Addition Example
This example illustrates how to use a RNN in tiny-dnn to perform sequential two-number addition, `i.e.`, 
``y[t+1] = x[t] + x[t+1]``

## Build and usage
Make sure cmake is set to build examples:
``cd build && cmake -DBUILD_EXAMPLES=Yes -DBUILD_TESTS=Yes -DCMAKE_BUILD_TYPE=Release .. && cmake -j4``

To execute (assumed to be in ./build):
./examples/example_recurrent_addition

## Overview
Basicaly, we need to define a ``recurrent_layer``, that will manage the state transitions of a 
given recurrent cell like ``gru``, ``lstm``, or ``rnn``:

```c++
void construct_net(N &nn,
                   const std::string rnn_type,
                   tiny_dnn::core::backend_t backend_type) {
  using fc              = tiny_dnn::fully_connected_layer;
  using tanh            = tiny_dnn::tanh_layer;
  using recurrent       = tiny_dnn::recurrent_layer;
  const int hidden_size = 100;  // recurrent state size
  const int seq_len     = 2;    // amount of numbers to add

  if (rnn_type == "rnn") {
    nn << recurrent(tiny_dnn::rnn(1, hidden_size), seq_len);
  } else if (rnn_type == "gru") {
    nn << recurrent(tiny_dnn::gru(1, hidden_size), seq_len);
  } else if (rnn_type == "lstm") {
    nn << recurrent(tiny_dnn::lstm(1, hidden_size), seq_len);
  }
  nn << tanh() << fc(hidden_size, 1);
}
```

* Since we are interested in adding **2** numbers, ``seq_len`` is set to 2.
* The ``hidden_size`` is the size of the recurrent state.
* The last ``fc`` layer, projects the output of the ``recurrent_layer`` to a single "desired" number.

Now, we need a dataset. Note that ``recurrent_layer`` requires data to be in the form: 
``(n sequences, sequence length, batch_size)``. This means that for: 1+2, 3+4, 5+6, 7+8, with a ``batch_size`` of 2
the data would be 13245768. This is done in ``gen_dataset``:

```c++
  for (int i = 0; i < n_samples; i++) {
    input.push_back({tiny_dnn::uniform_rand<float_t>(-1, 1)});
  }
  // Fill output in order n_samples * seq_len * batch-size
  output.resize(input.size());
  int n_seqs = n_samples / (seq_len * batch_size);
  for (int n = 0; n < n_seqs; n++) {
    int seq_start = n * batch_size * seq_len;
    for (int s = 0; s < seq_len; s++) {
      int item0_start = seq_start + (s - 1) * batch_size;  // x[t]
      int item1_start = seq_start + s * batch_size;        // x[t+1]
      for (int b = 0; b < batch_size; b++) {
        // y[0] = x[0]
        // y[t+1] = x[t] + x[t+1]
        output[item1_start + b].push_back(
          (s == 0 ? input[item1_start + b][0]
                  : input[item1_start + b][0] + input[item0_start + b][0]));
      }
    }
  }
```

Note that since we are adding two numbers, there is no ``x[-1]`` so that ``y[0] = x[-1] + x[0]``. Thus we start the output
sequence with ``y[0]=x[0]``.

Next, we define the training loop. As it can be seen inside the train function, batch_size is not used anymore but:
``int total_length = n_minibatch * seq_len;``
This is because recurrent layers need seq_len * batch_size samples at each step, and these must be 
also processed independently by the rest of non-recurrent layers.

Finally, we use `MSE` regression to train the model:
```c++
  // training
  nn.fit<tiny_dnn::mse>(optimizer, dataset[0], dataset[1], total_length,
                        n_train_epochs, on_enumerate_minibatch,
                        on_enumerate_epoch);
```

## Output
After training for some epochs, the code will enter in "demo mode". There you can continuously input
two numbers and the recurrent model will perform an approximate addition:

```
Input  numbers between -1 and 1.
Input number 1: 0.1
Input number 2: 0.4
Sum: 0.514308
Input number 1: 0.6
Input number 2: -0.9
Sum: -0.299533
Input number 1: 1.0
Input number 2: 1.0
Sum: 1.91505 # performance is worse at the extremes
Input number 1: 0
Input number 2: 0
Sum: 0.00183523
```

In the demo section, it is important to note that 
```c++
  layer->seq_len(1);
  layer->bptt_max(2);
```

Allows the rnn to be fed one input at a time while still remembering up to ``bptt_max`` steps.


