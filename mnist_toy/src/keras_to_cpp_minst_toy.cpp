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
#include "NnLayer.h"
#include "dumped.h"
#include "Utilities.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

int main(int argc, char** argv) {
	Clocks clk;
	vector_2d result;

	int iterations = 50;
	for(int i = 0; i < iterations; i++) {

		clk.start_Init();
		NeuralNetwork nn;
		nn.add_layer(new Flatten());
		nn.add_layer(new Dense(layer1DenseWeights, layer1DenseBias, layer1DenseActivation ));
		nn.add_layer(new Dense(layer2DenseWeights, layer2DenseBias, layer2DenseActivation ));
		nn.add_layer(new Dense(layer3DenseWeights, layer3DenseBias, layer3DenseActivation ));

		clk.start_FileRead();
		vector_2d input = Utilities::read_from_file("sample_mnist.dat");

		clk.start_Prediction();
		result = nn.predict(input);
		clk.iteration_finished();
	}

	int maxElementIndex = std::max_element(result.begin(), result.end()) - result.begin();
	cout << "Prediction: " << maxElementIndex << endl;
	for(size_t i = 0; i < result.size(); i++)
		cout << result[i][0] << " ";
	cout << endl << endl;

	cout << "Init:           " << clk.get_Init_average()          << " [ms]\t" << clk.get_Init_percent()		<< " %" << endl;
	cout << "File read:      " << clk.get_FileRead_average()      << " [ms]\t" << clk.get_Float_percent()       << " %" << endl;
	cout << "Calc out:       " << clk.get_Prediction_average()    << " [ms]\t" << clk.get_Prediction_percent()  << " %" << endl;
	cout << "Execution time: " << clk.get_Execuition_average()    << " [ms]"   << endl;

	return 0;
}
