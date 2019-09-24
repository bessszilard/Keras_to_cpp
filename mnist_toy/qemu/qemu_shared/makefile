CC = arm-linux-gnueabi-g++
# CC = g++
CXX_FLAGS = -std=c++11 -static
SRC_DIR = ../../src/

keras_to_cpp_minst_toy: keras_to_cpp_minst_toy.o NnLayer.o Utilities.o
	$(CC) keras_to_cpp_minst_toy.o NnLayer.o Utilities.o -o keras_to_cpp_minst_toy $(CXX_FLAGS) 

keras_to_cpp_minst_toy.o: $(SRC_DIR)keras_to_cpp_minst_toy.cpp
	$(CC) -c $(SRC_DIR)keras_to_cpp_minst_toy.cpp $(CXX_FLAGS)

NnLayer.o: $(SRC_DIR)NnLayer.cpp
	$(CC) -c $(SRC_DIR)NnLayer.cpp $(CXX_FLAGS)
	
Utilities.o: $(SRC_DIR)Utilities.cpp
	$(CC) -c $(SRC_DIR)Utilities.cpp $(CXX_FLAGS)
	
clean:
	rm *.o keras_to_cpp_minst_toy