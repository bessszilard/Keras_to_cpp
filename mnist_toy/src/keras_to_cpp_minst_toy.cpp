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

//using namespace std;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

vector<float> read_1d_array(ifstream &fin, int cols);
void read_from_file(const std::string &fname);
Matrix<float> Flatten_func(const Matrix<float> &input);
Matrix<float> relu_activation(const Matrix<float> &input);
Matrix<float> softmax_activation(const Matrix<float> &input);

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

//	Flatten layer0(1);
//	Dense layer1(layer1DenseWeights, layer1DenseBias, "relu"    );
//	Dense layer3(layer3DenseWeights, layer3DenseBias, "relu" 	);
//	Dense layer5(layer5DenseWeights, layer5DenseBias, "softmax" );

	NeuralNetwork nn(1);
	nn.add_layer(new Flatten(1));
	nn.add_layer(new Dense(layer1DenseWeights, layer1DenseBias, "relu"		));
	nn.add_layer(new Dense(layer3DenseWeights, layer3DenseBias, "relu" 		));
	nn.add_layer(new Dense(layer5DenseWeights, layer5DenseBias, "softmax" 	));
	clkInit.end = clock();

	clkFileRead.start = clkInit.end;
	read_from_file("sample_mnist.dat");
	Matrix<float> input(data[0]);
	clkFileRead.end = clock();

	clkCaclOut.start = clkFileRead.end;
//	Matrix<float> flat = layer0.get_output(input);//Flatten_func(input);
//	Matrix<float> layer2_Out = layer1.get_output(flat);
//	Matrix<float> layer4_Out = layer3.get_output(layer2_Out);
//	Matrix<float> layer6_Out = layer5.get_output(layer4_Out);
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

Matrix<float> Flatten_func(const Matrix<float> &input) {
	int orig_row = input.getHeight();
	int orig_col = input.getWidth();
	Matrix<float> flat(orig_row * orig_col, 1);

	for(int i = 0; i < orig_row; ++i) {
		for(int j = 0; j < orig_col; ++j) {
			int offset = (i * orig_col + j);
			flat.put(offset, 0, input.get(i, j));
		}
	}
	return flat;
}

