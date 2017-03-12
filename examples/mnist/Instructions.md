<p align="center"><b>Setup, build and run tinyddn on various OS</b><p align="center">

<b>tiny-dnn</b> is a C++11 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.


<p align="center"><b>How to Install/Build TinyDnn On Mac/Linux</b></p>
<b>Step:1</b> Install Cmake from https://cmake.org/install/  
<b>Step:2</b> Open Terminal (Press CMD+SpaceBar Enter Terminal)    
<b>Step:3</b>
1. sudo yum install git-all #(For Linux),Install https://sourceforge.net/projects/git-osx-installer/files/ (For mac)
2. cd ~    
3. git clone https://github.com/tiny-dnn/tiny-dnn.git
4. cd tiny-dnn
5. mkdir build
6. cd build
7. cmake .. -DBUILD_EXAMPLES=ON 
8. make
9. cd examples/
10. cd mnist/
11. Download [Mnist database](http://yann.lecun.com/exdb/mnist/) and copy to this directory  
12. Cd terminal to this directory and type g++ -std=c++11 train.cpp -DCNN_USE_OMP=0 -I ~/Desktop/tiny-dnn-master/  
13. In terminal type ./a.out .  
14. Now training will start.
15. Type g++ -std=c++11 -DCNN_USE_OMP=0 -I ~/Desktop/tiny-dnn-master/ test.cpp -DDNN_USE_IMAGE_API  
16. In terminal type ./a.out [4.bmp] (https://github.com/tiny-dnn/tiny-dnn/wiki/4.bmp)
 
Now you have succesfully installed tinydnn and also trained a mnist model.
<p align="center"><b>Sky is the limit.Try adding more layers and improve the accuracy</b><p align="center">



