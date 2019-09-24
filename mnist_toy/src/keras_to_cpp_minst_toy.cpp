//============================================================================
// Name        : keras_to_cpp_minst_toy.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <stdexcept>
#include "NnLayer.h"
#include "dumped.h"
#include "Utilities.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::vector;
using std::string;

int main(int argc, char** argv) {
	Clocks clk;
	vector_2d result;

	if(argc < 2)
		throw std::invalid_argument("Image path is needed");

	string imgPath = argv[1];
	cout << "Input: " << imgPath << endl;

	int iterations = 1;//10;
	for(int i = 0; i < iterations; i++) {

		clk.start_Init();
		NeuralNetwork nn;

		// if weights were not given through the console
		if(argc > 2) {
			nn.load_weights(argv[2]);
		}
		else {
			nn.add_layer(new Flatten());
			nn.add_layer(new Dense(layer1DenseWeights, layer1DenseBias, layer1DenseActivation ));
			nn.add_layer(new Dense(layer2DenseWeights, layer2DenseBias, layer2DenseActivation ));
			nn.add_layer(new Dense(layer3DenseWeights, layer3DenseBias, layer3DenseActivation ));
		}

		clk.start_FileRead();
//		vector_2d input = Utilities::read_from_file(imgPath);
//		vector_2d input = Utilities::read_from_binary_file(imgPath);

//		imgPath = "sample_mnist.dat";
//		vector_2d input1 = Utilities::read_from_file(imgPath);

		imgPath = "sample_mnist_bin.dat";
		vector_2d input = Utilities::read_from_binary_file(imgPath);


		clk.start_Prediction();
		result = nn.predict(input);
		clk.iteration_finished();
	}

	int maxElementIndex = std::max_element(result.begin(), result.end()) - result.begin();
	cout << "Prediction: " << maxElementIndex << endl << endl;
	for(size_t i = 0; i < result.size(); i++)
		cout << result[i][0] << " ";
	cout << endl;

	cout << "Init:           " << clk.get_Init_average()          << " [ms]\t" << clk.get_Init_percent()		<< " %" << endl;
	cout << "File read:      " << clk.get_FileRead_average()      << " [ms]\t" << clk.get_Float_percent()       << " %" << endl;
	cout << "Prediction out: " << clk.get_Prediction_average()    << " [ms]\t" << clk.get_Prediction_percent()  << " %" << endl;
	cout << "Execution time: " << clk.get_Execuition_average()    << " [ms]"   << endl;

	return 0;
}
