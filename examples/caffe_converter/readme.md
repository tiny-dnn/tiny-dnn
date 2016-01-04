# Import Caffe Model to tiny-cnn
tiny-cnn can import Caffe's trained models.

## Prerequisites for this example
- Google protobuf
- OpenCV

## Build

1 Use ```protoc``` to generte caffe.pb.cc and caffe.pb.h.
```bash
cd tiny_cnn/io/caffe
protoc caffe.proto --cpp_out=./
```

2 Compile ```tiny_cnn/io/caffe/caffe.pb.cc``` and ```examples/caffe_converter/caffe_converter.cpp``` and link them.

## Usage
```bash
./caffe_converter.bin [model-file] [trained-file] [mean-file] [label-file] [img-file]
```

In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_converter.bin\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel\
 imagenet_mean.binaryproto\
 synset_words.txt\
 cat.jpg
```

## Restrictions
- tiny-cnn's converter only supports single input/single output network without branch.