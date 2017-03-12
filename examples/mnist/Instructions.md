<p align="center"><b>Setup, build and run tiny-ddn on various OS</b><p align="center">

<b>tiny-dnn</b> is a C++11 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.


<b>How to Install/Build Tiny-dnn On Mac/Linux</b>  
<b>Step:1</b> Install Cmake from https://cmake.org/install/  
<b>Step:2</b> Open Terminal (Press CMD+SpaceBar Enter Terminal)    
<b>Step:3</b>

1. cd ~  
2. git clone https://github.com/tiny-dnn/tiny-dnn.git  
3. cd tiny-dnn  
4. mkdir build  
5. cd build  
6. cmake .. -DBUILD_EXAMPLES=ON  
7. make  
8. cd examples/  
9. ./example_mnist_train ~/tiny-dnn/data/  
10. Now training will start.  
11. After training is done,in terminal type ./example_mnist_test [4.bmp] (https://github.com/tiny-dnn/tiny-dnn/wiki/4.bmp)  
 
Now you have succesfully installed tinydnn and also trained a mnist model.  
<p align="center"><b>Sky is the limit.Try adding more layers and improve the accuracy</b><p align="center">  



