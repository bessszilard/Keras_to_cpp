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
#include "NnLayer.h"
#include "dumped.h"
#include "Matrix.h"

using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

vector<float> read_1d_array(ifstream &fin, int cols);
void read_from_file(const std::string &fname);

struct clock_bounds {
	clock_t start;
	clock_t end;
};

double getms(clock_bounds clk) {
	return ((double) (clk.end - clk.start) * 1000 / CLOCKS_PER_SEC);
}

int getpercent(clock_bounds clk, clock_bounds overall) {
	return 100 * getms(clk) / getms(overall);
}

int main() {
	clock_bounds clkExecuition, clkInit, clkFileRead, clkCaclOut;
	clkExecuition.start = clock();
	clkInit.start = clkExecuition.start; //clock();

	NeuralNetwork nn;
	nn.add_layer(new Flatten());
	nn.add_layer(new Dense(layer1DenseWeights, layer1DenseBias, "relu"		));
	nn.add_layer(new Dense(layer3DenseWeights, layer3DenseBias, "relu" 		));
	nn.add_layer(new Dense(layer5DenseWeights, layer5DenseBias, "softmax" 	));
	clkInit.end = clock();

	clkFileRead.start = clkInit.end;
	read_from_file("sample_mnist.dat");
	Matrix<float> input(data[0]);
	clkFileRead.end = clock();

	clkCaclOut.start = clkFileRead.end;
	Matrix<float> layer6_Out = nn.predict(input);
	clkCaclOut.end = clock();

	cout << layer6_Out.transpose() << endl;
	clkExecuition.end = clock();

	cout << "Init:           " << getms(clkInit) 		<< " [ms]\t" << getpercent(clkInit     , clkExecuition) << "%"<< endl;
	cout << "File read:      " << getms(clkFileRead)	<< " [ms]\t" << getpercent(clkFileRead , clkExecuition) << "%"<< endl;
	cout << "Calc out:       " << getms(clkCaclOut)		<< " [ms]\t" << getpercent(clkCaclOut  , clkExecuition) << "%"<< endl;
	cout << "Execution time: " << getms(clkExecuition)	<< " [ms]" << endl;

	return 0;
}

vector<float> read_1d_array(ifstream &fin, int cols) {
  vector<float> arr;
  arr.reserve(cols);
  float tmp_float;
  char tmp_char;
  fin >> tmp_char;
  for(int n = 0; n < cols; ++n) {
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
		vector < vector<float> > tmp_single_depth;
		for (int r = 0; r < m_rows; ++r) {
			vector<float> tmp_row = read_1d_array(fin, m_cols);
			tmp_single_depth.push_back(tmp_row);
		}
		data.push_back(tmp_single_depth);
	}
	fin.close();
}
