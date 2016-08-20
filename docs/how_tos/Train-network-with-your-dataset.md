# Train network with your original dataset
Here are some examples.

### 1. using opencv (image file => vec_t)

```cpp
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

// convert image to vec_t
void convert_image(const std::string& imagefilename,
                   double scale,
                   int w,
                   int h,
                   std::vector<vec_t>& data)
{
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));
    vec_t d;

    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) { return c * scale; });
    data.push_back(d);
}

// convert all images found in directory to vec_t
void convert_images(const std::string& directory,
                    double scale,
                    int w,
                    int h,
                    std::vector<vec_t>& data)
{
    path dpath(directory);

    BOOST_FOREACH(const path& p, 
                  std::make_pair(directory_iterator(dpath), directory_iterator())) {
        if (is_directory(p)) continue;
        convert_image(p.string(), scale, w, h, data);
    }
}
```

Another example can be found in [issue#16](https://github.com/tiny-dnn/tiny-dnn/issues/16), which can treat color channels.

### 2. using [mnisten](https://github.com/nyanp/mnisten) (image file => idx format)
mnisten is a library to convert image files to idx format.
```
mnisten -d my_image_files_directory_name -o my_prefix -s 32x32
```
After generating idx files, you can use parse_mnist_images / parse_mnist_labels utilities in mnist_parser.h

### 3. from levelDB (caffe style => [vec_t, label_t])
[Caffe](https://github.com/BVLC/caffe/) supports levelDB data format. Following code can convert levelDB created by Caffe into data/label arrays.

```cpp
#include "leveldb/db.h"

void convert_leveldb(const std::string& dbname,
                     double scale,
                     std::vector<vec_t>& data,
                     std::vector<label_t>& label)
{
    leveldb::DB *db;
    leveldb::Options options;
    options.create_if_missing = false;
    auto status = leveldb::DB::Open(options, dbname, &db);

    leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        const char* src = it->value().data();
        size_t sz = it->value().size();
        vec_t d;
        std::transform(src, src + sz - 1, std::back_inserter(d),
                       [=](char c){ return c * scale; });
        data.push_back(d);
        label.push_back(src[sz - 1]);
    }
    delete it;
    delete db;
}
```