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