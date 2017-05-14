# ImageNet 1000 classes classification example using AlexNet

[ImageNet-1000] is a common dataset 
for object classification.
The problem is to classify 227x227 RGB (thus 227x227x3=154,587 dimensions) image into 1000 classes 

Download the pretrained AlexNet model for Tiny-DNN:
https://drive.google.com/file/d/0B4zhKe-HzGEuX0FpNThhUmlsSlE/view?usp=sharing

Example usage:
```
./example_imagenet_test.bin\
 tiny-model-alexnet\
 cat.jpg
```