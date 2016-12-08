#ifndef CNN_NO_SERIALIZATION

CNN_REGISTER_LAYER_WITH_ACTIVATIONS(convolutional_layer, conv);
CNN_REGISTER_LAYER_WITH_ACTIVATIONS(fully_connected_layer, fully_connected);
CNN_REGISTER_LAYER_WITH_ACTIVATIONS(average_pooling_layer, avepool);
CNN_REGISTER_LAYER_WITH_ACTIVATIONS(max_pooling_layer, maxpool);
CNN_REGISTER_LAYER_WITH_ACTIVATIONS(linear_layer, linear);
CNN_REGISTER_LAYER_WITH_ACTIVATIONS(lrn_layer, lrn);

CNN_REGISTER_LAYER(batch_normalization_layer, batchnorm);
CNN_REGISTER_LAYER(concat_layer, concat);
CNN_REGISTER_LAYER(dropout_layer, dropout);
CNN_REGISTER_LAYER(power_layer, power);
CNN_REGISTER_LAYER(slice_layer, slice);
CNN_REGISTER_LAYER(elementwise_add_layer, elementwise_add);

#endif
