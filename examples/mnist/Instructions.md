<p align="center"><b>Setup, build and run tiny-dnn on various operating systems</b><p align="center">

<b>tiny-dnn</b> is a C++11 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.


<b>How to Install/Build Tiny-dnn On Mac/Linux</b>  
<b>Step:1</b> Install Cmake from https://cmake.org/install/  
<b>Step:2</b> Open Terminal (Press CMD+SpaceBar Enter Terminal)    
<b>Step:3</b>
```
1. cd ~/install_path # adjust `install_path` where you want to install tiny-dnn
2. git clone https://github.com/tiny-dnn/tiny-dnn.git  
3. cd tiny-dnn  
4. mkdir build & cd build   
5. cmake -DBUILD_EXAMPLES=ON .. & make  
6. cd examples   
7. ./example_mnist_train ../../data  
```
# Now training will start.  
# After training is done, in terminal type  
```
8. ./example_mnist_test sample_img.bmp   
```
Now you have succesfully installed tiny-dnn and also trained a mnist model.  



