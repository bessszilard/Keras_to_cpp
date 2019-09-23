#Toy MNIST model
##Task description
1. Implement the trained model for a Cortex A53 processor in C or C++. You can only use standard libraries.
2. Deliver the application as source code and build scripts and the resulting binary that can run in qemu simulator.
3. The application must accept input files of 28x28 bytes containing MNIST handwritten digits and must output the predicted digit on the console and the execution time
4. Demonstrate that you have actually run this application on an ARM CPU or in a simulator..
5. Outline your ideas on improving this model in speed and accuracy

##1. Implement the trained model for a Cortex A53 processor in C or C++. You can only use standard libraries.

alma


##2. Deliver the application as source code and build scripts and the resulting binary that can run in qemu simulator.

###2.1. Set up Qemu

* Download AARCH64 build root image [aarch64-linux-3.15rc2-buildroot.img](https://github.com/675816156/Qemu-aarch64) 

* Run the virtual machine: [source]((https://www.bennee.com/~alex/blog/2014/05/09/running-linux-in-qemus-aarch64-system-emulation-mode/)) ```
$ qemu-system-aarch64 -machine virt -cpu cortex-a53 -machine type=virt -nographic -smp 1 -m 2048 -kernel aarch64-linux-3.15rc2-buildroot.img --append "console=ttyAMA0"```

* Share “/home/szilard/qemu/bennee/qemu_shared” folder with Qemu virtual machine:
```$ qemu-system-aarch64 -machine virt -cpu cortex-a53 -machine type=virt -nographic -smp 1 -m 2048 -kernel aarch64-linux-3.15rc2-buildroot.img --append "console=ttyAMA0" -fsdev local,id=r,path=/home/szilard/qemu/bennee/qemu_shared,security_model=none -device virtio-9p-device,fsdev=r,mount_tag=r ```

* Mount the shared folder:
``` $ mount -t 9p -o trans=virtio r /mnt ```

###2.2. Cross compiling for Cortex A53
* Compile a single file named helloworld.cpp
```$ arm-linux-gnueabi-g++ helloword.cpp -o helloword-arm-cpp -static```
* Creating a build scripts
```
CC = arm-linux-gnueabi-g++
# CC = g++
CFLAGS = -std=c++11 -static

keras_to_cpp_minst_toy: keras_to_cpp_minst_toy.o NnLayer.o
	$(CC) keras_to_cpp_minst_toy.o NnLayer.o -o keras_to_cpp_minst_toy $(CFLAGS)

keras_to_cpp_minst_toy.o: keras_to_cpp_minst_toy.cpp NnLayer.h
	$(CC) -c keras_to_cpp_minst_toy.cpp $(CFLAGS)

NnLayer.o: NnLayer.cpp NnLayer.h
	$(CC) -c NnLayer.cpp $(CFLAGS)	
	
clean:
	rm *.o keras_to_cpp_minst_toy
```

## 3. The application must accept input files of 28x28 bytes containing MNIST handwritten digits and must output the predicted digit on the console and the execution time

* Write image to binary file in python:
```
# store one sample in text file
with open("./sample_mnist.dat", "w") as fin:
    fin.write("1 28 28\n")
    a = x_train[3]
    for b in a:
        fin.write(str(b)+'\n')
```

* Read image from binary file in C++:
```
vector_1d read_1d_array(ifstream &fin, int cols) {
	vector_1d arr;
	arr.reserve(cols);
	float tmp_float;
	char tmp_char;
	fin >> tmp_char;
	for (int n = 0; n < cols; ++n) {
		fin >> tmp_float;
		arr.push_back(tmp_float);
	}
	fin >> tmp_char;
	return arr;
}

void read_from_file(const std::string &fname) {
	int m_depth, m_rows, m_cols;
	ifstream fin(fname.c_str());
	if(fin.fail())
		throw std::invalid_argument( "can't open " + fname);
	fin >> m_depth >> m_rows >> m_cols;
	data.reserve(m_depth * m_rows * m_cols);

	for (int d = 0; d < m_depth; ++d) {
		vector_2d tmp_single_depth;
		for (int r = 0; r < m_rows; ++r) {
			vector_1d tmp_row = read_1d_array(fin, m_cols);
			tmp_single_depth.push_back(tmp_row);
		}
		data.push_back(tmp_single_depth);
	}
	fin.close();
}
```

