//============================================================================
// Name        : keras_to_cpp_minst_toy.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Simple neural network, which classifies images from
//				 MNIST data set
//============================================================================

#include <iostream>
#include "NnLayer.h"
#include "dumped.h"
#include "Utilities.h"

using std::cout;
using std::endl;
using std::string;

int main(int argc, char** argv) {
	Clocks clk;
	int result;

	if(argc < 2)
		throw std::invalid_argument("Image path is needed");

	string imgPath = argv[1];
//	cout << "Input: " << imgPath << endl;

	int iterations = 50;
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
		vector_2d input = Utilities::read_from_binary_file(imgPath);

		clk.start_Prediction();
		result = nn.classify(input);
		clk.iteration_finished();
	}

	cout << "Predicted digit: " << result << endl;
	cout << endl;

	cout << "Init:         \t" << clk.get_Init_average()          << "\t[ms]\t" << clk.get_Init_percent()		 << "%" << endl;
	cout << "File read:    \t" << clk.get_FileRead_average()      << "\t[ms]\t" << clk.get_Float_percent()       << "%" << endl;
	cout << "Prediction:   \t" << clk.get_Prediction_average()    << "\t[ms]\t" << clk.get_Prediction_percent()  << "%" << endl;
	cout << "Whole process:\t" << clk.get_Execuition_average()    << "\t[ms]"   << endl;

	return 0;
}
