# Convert Caffe Model to tiny-dnn
tiny-dnn can convert Caffe's trained models.

## Prerequisites for this example
- Google protobuf
- OpenCV

## Build

1 Use ```protoc``` to generte caffe.pb.cc and caffe.pb.h.
```bash
cd tiny_dnn/io/caffe
protoc caffe.proto --cpp_out=./
```

2 Compile ```tiny_dnn/io/caffe/caffe.pb.cc``` and ```examples/caffe_convert_model/caffe_convert_model.cpp``` and link them.

## Usage
```bash
./caffe_converter.bin [model-file] [trained-file] [mean-file] [label-file] [img-file]
```

In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_convert_model.bin\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel
```

## Restrictions
- tiny-dnn's converter only supports single input/single output network without branch.