# Changing from v0.0.1
This section explains the API changes from v0.0.1.

## How to specify the loss and the optimizer 
In v0.0.1, the loss function and the optimization algorithm are treated as template parameter of ```network```.

```cpp
//v0.0.1
network<mse, adagrad> net;
net.train(x_data, y_label, n_batch, n_epoch);
```

From v0.1.0, these are treated as parameters of train/fit functions.

```cpp
//v0.1.0
network<sequential> net;
adagrad opt;
net.fit<mse>(opt, x_data, y_label, n_batch, n_epoch);
```

## Training API for regression
In v0.0.1, the regression and the classification have the same API:

```cpp
//v0.0.1
net.train(x_data, y_data, n_batch, n_epoch); // regression
net.train(x_data, y_label, n_batch, n_epoch); // classification
```

From v0.1.0, these are separated into ```fit``` and ```train```.

```cpp
//v0.1.0
net.fit<mse>(opt, x_data, y_data, n_batch, n_epoch); // regression
net.train<mse>(opt, x_data, y_label, n_batch, n_epoch); // classification
```

    
