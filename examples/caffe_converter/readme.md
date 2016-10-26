# Import Caffe Model to tiny-dnn
tiny-dnn can import Caffe's trained models.

## Prerequisites for this example
- Google protobuf
- OpenCV

## Build

1 Use ```protoc``` to generte caffe.pb.cc and caffe.pb.h.
```bash
cd tiny_dnn/io/caffe
protoc caffe.proto --cpp_out=./
```

2 Compile ```tiny_dnn/io/caffe/caffe.pb.cc``` and ```examples/caffe_converter/caffe_converter.cpp``` and link them.

## Usage
```bash
./caffe_converter.bin [mode] [model-file] [trained-file] [mean-file] [label-file] [mode-file]
```
Where **mode** can be either test or archive

### Testing a Pre-trained Model
In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_converter.bin\
 test\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel\
 imagenet_mean.binaryproto\
 synset_words.txt\
 cat.jpg
```

### Archiving a Pre-trained Model
In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_converter.bin\
 archive\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel\
 imagenet_mean.binaryproto\
 synset_words.txt\
 bvlc_reference_caffenet.tinydnn
```

The archived file contains the converted tinydnn network, the mean file and the labels.  It can be used by at a later time (without caffe/protobuf dependencies) by doing something like,

```
network<sequential> net;
image<float> mean;
std::vector<std::string> labels;

std::ifstream ifs("bvlc_reference_caffenet.tinydnn", std::ios::binary | std::ios::in);
cereal::BinaryInputArchive bi(ifs);

try {
    net.from_archive(bi);

    size_t w, h, depth;
    image_type type;
    std::vector<float> data;

    bi(cereal::make_nvp("width", w),
       cereal::make_nvp("height", h),
       cereal::make_nvp("depth", depth),
       cereal::make_nvp("type", type),
       cereal::make_nvp("data", data),
       cereal::make_nvp("labels", labels)
       );

    mean = image<float>(shape3d(w, h, depth), type);
    mean.from_rgb(data.begin(), data.end());

} catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
}
// use the net, mean and labels as shown in the caffe_converter.cpp 
```
## Restrictions
- tiny-dnn's converter only supports single input/single output network without branch.