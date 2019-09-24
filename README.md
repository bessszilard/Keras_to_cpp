# Toy MNIST model
## Task description
1. Implement the trained model for a Cortex A53 processor in C or C++. You can only use standard libraries.
2. Deliver the application as source code and build scripts and the resulting binary that can run in qemu simulator.
3. The application must accept input files of 28x28 bytes containing MNIST handwritten digits and must output the predicted digit on the console and the execution time
4. Demonstrate that you have actually run this application on an ARM CPU or in a simulator..
5. Outline your ideas on improving this model in speed and accuracy

## Project structure

```
├── mnist_toy
│   ├── Debug
│   │   ├── makefile
│   │   ├── objects.mk
│   │   ├── sources.mk
│   │   └── src
│   │       └── subdir.mk
│   ├── dumped.nnet
│   ├── nn_file_gen
│   │   ├── h5_to_dumped_h.py
│   │   ├── h5_to_dumped_nnet.py
│   │   └── mnist_toy_model.ipynb
│   ├── qemu
│   │   ├── aarch64-linux-3.15rc2-buildroot.img
│   │   └── qemu_shared
│   │       ├── dumped.nnet
│   │       ├── makefile
│   │       ├── sample_mnist_bin.dat
│   │       └── sample_mnist.dat
│   ├── sample_mnist_bin.dat
│   ├── sample_mnist.dat
│   └── src
│       ├── dumped.h
│       ├── keras_to_cpp_minst_toy.cpp
│       ├── NnLayer.cpp
│       ├── NnLayer.h
│       ├── nnVector.h
│       ├── Utilities.cpp
│       └── Utilities.h
├── README.md
└── README.pdf
```
# Summary




## 1. Implement the trained model for a Cortex A53 processor in C or C++. You can only use standard libraries.

### 1.1. Generate JSON, and store weights
```
# store model
with open('./my_nn_arch.json', 'w') as fout:
    fout.write(model.to_json())
model.save_weights('./my_nn_weights.h5', overwrite=True)
```
### 1.2. Generate plain text and header files
```
$ python h5_to_dumped_h.py -a my_nn_arch.json -w my_nn_weights.h5 -o ../dumped.nnet -v 1
$ python h5_to_dumped_nnet.py -a my_nn_arch.json -w my_nn_weights.h5 -o ../src/dumped.h -v 1
```
### 1.3. Importing generated files
The program can work in 2 modes, in *modifible* and *fixed* weights. In *modifible weights* mode, the program reads the neural network weights from an exteral .nnet file, which can be given as an argument in the command line. In this case, the neural network architure and weights is determined in the external file. The architecture and weight loading process is handeled in ```NeuralNetwork::load_weights()``` function in [NnLayer.cpp](mnist_toy/src/NnLayer.cpp) file. 
In the fixed weights mode neural network architecture is created manually in [keras_to_cpp_minst_toy.cpp](mnist_toy/src/keras_to_cpp_minst_toy.cpp) file in the main function, and weights are save in [dumped.h](mnist_toy/src/dumped.h) file. The adventage of this solution, that the innitialization process can be more than 120x faster.

### 1.4. Prediction
The whole prediction is done in ```NeuralNetwork::predict()``` function in [NnLayer.cpp](mnist_toy/src/NnLayer.cpp). For the generalization output and the input is also a vector_2d variable. The classified number is the of the output vector's biggest element. 
```
vector_2d NeuralNetwork::predict(const vector_2d &input) {
	vector_2d temp = input;
	for(auto layer : m_layers) {
		temp = layer->get_output(temp);
	}
	return temp;
}
```

## 2. Deliver the application as source code and build scripts and the resulting binary that can run in qemu simulator.

### 2.1. Set up Qemu

* Download AARCH64 build root image [aarch64-linux-3.15rc2-buildroot.img](https://github.com/675816156/Qemu-aarch64) 

* Run the virtual machine: [source]((https://www.bennee.com/~alex/blog/2014/05/09/running-linux-in-qemus-aarch64-system-emulation-mode/)) 
```
$ qemu-system-aarch64 -machine virt -cpu cortex-a53 -machine type=virt \
-nographic -smp 1 -m 2048 -kernel aarch64-linux-3.15rc2-buildroot.img  \
--append "console=ttyAMA0"
```

* Exit from Qemu console 
``` Ctrl-A X ```

* Share “/home/szilard/qemu/bennee/qemu_shared” folder with Qemu virtual machine:
```
$ qemu-system-aarch64 -machine virt -cpu cortex-a53 -machine type=virt \
 -nographic -smp 1 -m 2048 -kernel aarch64-linux-3.15rc2-buildroot.img  \
--append "console=ttyAMA0" \
-fsdev local,id=r,path=/home/szilard/Documents/git/Keras_to_cpp/mnist_toy/qemu/qemu_shared,security_model=none \
-device virtio-9p-device,fsdev=r,mount_tag=r
```

* Mount the shared folder:
``` $ mount -t 9p -o trans=virtio r /mnt ```

### 2.2. Cross compiling for Cortex A53
* Compile a single file named helloworld.cpp

```$ arm-linux-gnueabi-g++ helloword.cpp -o helloword-arm-cpp -static```

* [Build scripts file](mnist_toy/qemu/qemu_shared/makefile)

## 3. The application must accept input files of 28x28 bytes containing MNIST handwritten digits and must output the predicted digit on the console and the execution time

* Binary image reading is implemented in ```Utilities::read_from_binary_file()``` function in [Utilities.cpp](mnist_toy/src/Utilities.cpp). After the file reading 
