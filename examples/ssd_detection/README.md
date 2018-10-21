# Single Shot MultiBox Detector (SSD)

A TinyDNN implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

## Prerequisites for this example

- Download PyTorch pretrained models from [here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)
- Convert PyTorch models to TinyDNN with following commands

```
mkdir models
python convert_models.py /path/to/model models
```

## Use SSD in Object Detection

```
./example_ssd_test /folder/to/models/ /path/to/image
```

You will get output similar to following:

```
Bounding box coordinates:
x_min = 19.6578, x_max = 66.2639, y_min = 240.801, y_max = 270.333, class = 1, score = 0.989524
x_min = -3.73388, x_max = 302.883, y_min = 43.5821, y_max = 202.043, class = 1, score = 0.900133
x_min = 17.771, x_max = 54.9812, y_min = 237.505, y_max = 259.987, class = 1, score = 0.660543
```

If you see the following message:

```
Failed to load weights from models/01.weights
Failed to load weights from models/02.weights
...
Failed to load weights from models/18.weights
```

Please make sure the path of weight files.

## Detection Results

Here's an example of object detection results produced by SSD.

![Airplane](https://user-images.githubusercontent.com/1730504/47263759-67bc1900-d53a-11e8-91cd-4bb4648668b7.png)

![Sofa](https://user-images.githubusercontent.com/1730504/47264055-d9976100-d540-11e8-98a5-0af7871374fd.png)
