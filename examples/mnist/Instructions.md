<p align="center"><b>Setup, build and run tiny-dnn on Mac/Linux</b><p align="center">




<b>How to Install/Build Tiny-dnn On Mac/Linux</b>  
<b>Step:1</b> Install Cmake from https://cmake.org/install/  
<b>Step:2</b> Open Terminal.  
<b>Step:3</b>
```bash
cd ~/install_path # adjust `install_path` where you want to install tiny-dnn
git clone https://github.com/tiny-dnn/tiny-dnn.git  
cd tiny-dnn  
mkdir build & cd build   
cmake -DBUILD_EXAMPLES=ON .. & make  
cd examples   
./example_mnist_train ../../data  
```
Now training will start.  
After training is finished, in terminal type  
```
./example_mnist_test sample_img.bmp   
```
Now you have succesfully installed tiny-dnn and also trained a mnist model.  



