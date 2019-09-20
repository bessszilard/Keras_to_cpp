//============================================================================
// Name        : matrix_multp.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include "dumped.h"
#include "Matrix.h"
#include <math.h>
#include <time.h>

//using namespace std;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;
std::vector<std::vector<float> > x1 = { {1.0, 2.0, 3.0},
									    {1.0, 2.0, 3.0}};
std::vector<std::vector<float> > x2 = { {2.0, 4.0, 2.0},
									    {2.0, 2.0, 2.0}};

std::vector<std::vector<float> > x3 = { {3.0, 0.0, 0.0},
									    {0.0, 3.0, 0.0},
										{0.0, 0.0, 3.0}};
float xy1[][2] = {{1, 2}, {2, 3}};
float xy2[][2] = {{1, 2}, {2, 3}};
std::vector<std::vector<float> > x4 ={
{ 0.02233464,-0.0076446 , 0.06854594,-0.03888208, 0.01313791,-0.03845475,
  0.00761586,-0.0746503 ,-0.06191996,-0.01639963,-0.05626907, 0.06444412,
  0.06227136, 0.07696087, 0.01255471,-0.04042   ,-0.04759897, 0.05135382,
 -0.07894874, 0.0595108 , 0.00397626, 0.08103367, 0.0617558 ,-0.0647239,
 -0.08365198, 0.04439714,-0.00367419, 0.03721378, 0.0335324 ,-0.04241298,
 -0.0739582 , 0.06032496,-0.0797427 , 0.02541213,-0.04730907, 0.04404473,
  0.02509418, 0.03607042,-0.02824639,-0.02262121,-0.05503771, 0.06571131,
  0.05275082, 0.07119242,-0.07545521,-0.05785039, 0.02169149, 0.04614723,
 -0.06768519, 0.07200833, 0.02271841, 0.03529294, 0.00760138, 0.03720661,
  0.03490973, 0.05014687,-0.03183138, 0.07928092,-0.07114929, 0.05767576,
 -0.00332975,-0.07563288,-0.03433243,-0.06089015},
{-0.08008682, 0.04645916, 0.05700923,-0.02240796, 0.03004926, 0.00774903,
  0.06416834,-0.03794123, 0.01102622, 0.0514191 ,-0.06645077,-0.00946285,
  0.00587874, 0.02017628,-0.0055288 ,-0.06345221,-0.05530353, 0.03499863,
  0.02349353,-0.05567695,-0.05505501,-0.07996832,-0.01671235,-0.07777055,
 -0.0216837 ,-0.04115792,-0.06857993,-0.07739972, 0.00808586,-0.0599129,
  0.02203641,-0.03235224, 0.01681432,-0.03244704, 0.03151502,-0.00591373,
 -0.02061412,-0.08087355, 0.04730957, 0.07268084, 0.03897285, 0.04692654,
  0.06883518,-0.00942667, 0.0285142 ,-0.0305282 , 0.06048196,-0.07691203,
 -0.08130455, 0.00933205, 0.0579265 ,-0.06770211, 0.02627587,-0.04005055,
 -0.08224   ,-0.04977876, 0.0483787 , 0.07092553, 0.04590661, 0.04729255,
 -0.0209499 ,-0.02192922, 0.05343739,-0.00349886}
 };

std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

vector<float> read_1d_array(ifstream &fin, int cols);
void read_from_file(const std::string &fname);
Matrix<float> Flatten(const Matrix<float> &input);
Matrix<float> relu_activation(const Matrix<float> &input);
Matrix<float> softmax_activation(const Matrix<float> &input);
int main() {
//	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
//	Matrix<T>(std::vector<std::vector<T> > const &array);
	clock_t tStart = clock();

//	Matrix<float> a(x1);
	Matrix<float> b(x2);
//
//	Matrix<float> c = a + b;
//	Matrix<float> d = b * Matrix<float>(x3);
//
//	cout << c << endl;
//	cout << d << endl;
//
//	Matrix<float> e(x4);
//	cout << e.getHeight() << ", " << e.getWidth()  << endl;
	read_from_file("sample_mnist.dat");

	Matrix<float> layer1W(layer1DenseWeights);
	Matrix<float> layer3W(layer3DenseWeights);
	Matrix<float> layer5W(layer5DenseWeights);

	Matrix<float> layer1D(layer1DenseBias);
	Matrix<float> layer3D(layer3DenseBias);
	Matrix<float> layer5D(layer5DenseBias);

	Matrix<float> input(data[0]);

//	cout << layer1W.getHeight() << "\t" << layer1W.getWidth() << endl;
//	cout << layer3W.getHeight() << "\t" << layer3W.getWidth() << endl;
//	cout << layer5W.getHeight() << "\t" << layer5W.getWidth() << endl;
//
//	cout << layer1D.getHeight() << "\t" << layer1D.getWidth() << endl;
//	cout << layer3D.getHeight() << "\t" << layer3D.getWidth() << endl;
//	cout << layer5D.getHeight() << "\t" << layer5D.getWidth() << endl;
//
//	cout << input.getHeight() << "\t" << input.getWidth() << endl;

	Matrix<float> flat = Flatten(input);

//	cout << flat << endl;
	Matrix<float> layer1_Out = layer1W.transpose() * flat + layer1D.transpose();
	Matrix<float> layer2_Out = relu_activation(layer1_Out);

	Matrix<float> layer3_Out = layer3W.transpose() * layer2_Out + layer3D.transpose();
	Matrix<float> layer4_Out = relu_activation(layer3_Out);

	Matrix<float> layer5_Out = layer5W.transpose() * layer4_Out + layer5D.transpose();
	Matrix<float> layer6_Out = softmax_activation(layer5_Out);


	cout << layer6_Out.transpose() << endl;

//	cout << layer1_Out.getHeight() << "\t" << layer1_Out.getWidth() << endl;
//	cout << flat.transpose() << endl;
//	cout << layer1_Out.transpose() << endl;
//	Matrix<float> layer_next = layer1 *

//	Matrix<int> a,b,c;

//	int z;
	cout << endl << "Execution time: ";
	cout << ((double) (clock() - tStart) * 1000 / CLOCKS_PER_SEC) << "[ms]" << endl;

	return 0;
}

vector<float> read_1d_array(ifstream &fin, int cols) {
  vector<float> arr;
  float tmp_float;
  char tmp_char;
  fin >> tmp_char; // for '['
  for(int n = 0; n < cols; ++n) {
    fin >> tmp_float;
    arr.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'
  return arr;
}

void read_from_file(const std::string &fname) {

	int m_depth, m_rows, m_cols;
	ifstream fin(fname.c_str());
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

Matrix<float> Flatten(const Matrix<float> &input) {
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


Matrix<float> relu_activation(const Matrix<float> &input) {
	if(input.getWidth() != 1)
		throw std::invalid_argument("This is not a column vector L185");

	int orig_row = input.getHeight();
	Matrix<float> result(orig_row, 1);
	for(int i = 0; i < orig_row; ++i) {
		float temp = input.get(i, 0);
		if (temp < 0)
			temp = 0;
		result.put(i, 0, temp);
	}
	return result;
}

Matrix<float> softmax_activation(const Matrix<float> &input) {
	float sum = 0.0;
	if(input.getWidth() != 1)
		throw std::invalid_argument("This is not a column vector");

	int orig_row = input.getHeight();
	Matrix<float> result(orig_row, 1);

	for(int k = 0; k < orig_row; ++k) {
	  float temp = exp(input.get(k, 0));
	  result.put(k, 0, temp);
	  sum += temp;
	}
	result /= sum;
//	for(int k = 0; k < orig_row; ++k) {
//	  y[k] /= sum;
//	}
	return result;
}
