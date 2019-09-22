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
	float t_execution = 0;
	float t_Init = 0;
	float t_FileRead = 0;
	float t_CalcOut = 0;
	Matrix<float> result;
	int iterations = 50;
	for(int i = 0; i < iterations; i++) {
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
		result = nn.predict(input);
		clkCaclOut.end = clock();

		clkExecuition.end = clock();
		t_Init      += getms(clkInit) 		;
		t_FileRead  += getms(clkFileRead)	;
		t_CalcOut   += getms(clkCaclOut)	;
		t_execution += getms(clkExecuition) ;

	}
	cout << result.transpose() << endl;

	cout << "Init:           " << t_Init      / iterations << " [ms]\t" << (int)(100 * (t_Init      / t_execution)) << "%"<< endl;
	cout << "File read:      " << t_FileRead  / iterations << " [ms]\t" << (int)(100 * (t_FileRead  / t_execution)) << "%"<< endl;
	cout << "Calc out:       " << t_CalcOut   / iterations << " [ms]\t" << (int)(100 * (t_CalcOut   / t_execution)) << "%"<< endl;
	cout << "Execution time: " << t_execution / iterations << " [ms]"   << endl;

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
