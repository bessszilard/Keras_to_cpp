CC = arm-linux-gnueabi-g++
# CC = g++
CXXFLAGS = -std=c++11 -static
SRCDIR = ../../src/

keras_to_cpp_minst_toy: keras_to_cpp_minst_toy.o NnLayer.o
	$(CC) keras_to_cpp_minst_toy.o NnLayer.o -o keras_to_cpp_minst_toy $(CXXFLAGS)

keras_to_cpp_minst_toy.o: $(SRCDIR)keras_to_cpp_minst_toy.cpp $(SRCDIR)NnLayer.h
	$(CC) -c $(SRCDIR)keras_to_cpp_minst_toy.cpp $(CXXFLAGS)

NnLayer.o: $(SRCDIR)NnLayer.cpp $(SRCDIR)NnLayer.h
	$(CC) -c $(SRCDIR)NnLayer.cpp $(CXXFLAGS)	
	
clean:
	rm *.o keras_to_cpp_minst_toy