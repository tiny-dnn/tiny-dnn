<p align="center"><b>How to build MNIST example programs.</b><p align="center">



<b>Step:1</b> Install CMake from https://cmake.org/install/  
<b>Step:2</b> Open Terminal.  
<b>Step:3</b>
```bash
cd install_path # adjust `install_path` to where you want to install tiny-dnn
git clone https://github.com/tiny-dnn/tiny-dnn.git  
cd tiny-dnn  
mkdir build & cd build   
CMake -DBUILD_EXAMPLES=ON .. & make  
cd examples   
./example_mnist_train ../../data  
```
Now training will start.  
After training is finished, in terminal type  
```
./example_mnist_test IMAGE_TO_BE_TESTED.bmp 
```
Now you have successfully trained and tested a MNIST model. mnist model.  



