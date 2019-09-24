/*
 * NnLayer.cpp
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#include <math.h>
#include <fstream>
#include <time.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "NnLayer.h"

using std::cout;
using std::endl;

// Dense ==========================================================================================
// private ----------------------------------------------------------------------------------------
/*
 * Return the activation function result
 * The input and the output is is vector_2d to get compatibility with Flatten's get_output function
 */
vector_2d Dense::activaction(const vector_2d &input) {
	if (input[0].size() != 1)
		throw std::invalid_argument(
				"Activation couldn't completed, because this is not a column vector");
	int orig_row = input.size();
	vector_2d result(orig_row, vector_1d(1));

	if (m_activation_type == "relu") {
		for (int i = 0; i < orig_row; ++i)
			result[i][0] = input[i][0] < 0 ? 0 : input[i][0];
		return result;
	}
	else if (m_activation_type == "softmax") {
		nn_cal_type sum = 0.0;
		for (int k = 0; k < orig_row; ++k) {
			nn_cal_type temp = exp(input[k][0]);
			result[k][0] = temp;
			sum += temp;
		}
		for (int k = 0; k < orig_row; k++)
			result[k][0] /= sum;
		return result;
	} else {
		throw std::invalid_argument("Unknown activation function");
	}
}

// public ---------------------------------------------------------------------------------------
void Dense::dotprod(vector_2d &m_weights, int col, const vector_2d &input, nn_cal_type m_bias, std::string act_type, nn_cal_type &output) {
	nn_cal_type sum = 0.0f;
	for (size_t k = 0; k < m_weights.size(); ++k) {
		sum += m_weights[k][col] * input[k][0];
	}
	sum += m_bias;
	output = sum;
}

Dense::Dense(const std::vector<std::vector<nn_cal_type> > &weights,
				 const std::vector<std::vector<nn_cal_type> > &bias,
				 const std::string &a_type) :
	m_weights(weights), m_bias(bias), m_activation_type(a_type)
{
	if(1 < m_bias.size())
		throw std::invalid_argument("Invalid bias size");
}

Dense::~Dense() {}

void Dense::load_weights(std::ifstream &fin) {
	int input_cnt, neurons;
	fin >> input_cnt >> neurons;
	set_inputSize(input_cnt);
	set_outputSize(neurons);

	char tmp_char = ' ';
	std::string temp_str;

	m_weights = vector_2d(input_cnt, vector_1d(neurons));
	for (int i = 0; i < input_cnt; ++i) {
		fin >> tmp_char; // for '['
		for (int n = 0; n < neurons; ++n)
			fin >> m_weights[i][n];
		fin >> tmp_char; // for ']'
	}

	m_bias = vector_2d(1, vector_1d(neurons));
	fin >> tmp_char; // for '['
	for (int n = 0; n < neurons; ++n)
		fin >> m_bias[0][n];
	fin >> tmp_char; // for ']'

	fin >> temp_str >> tmp_char >> temp_str >> temp_str >> m_activation_type;	// layer 1 Dense activation relu
}

/*
 * Calculates the output of the Dense layer.
 * Formula: transpose(W) * X + transpose(B)
 *		(n, 1) =  (n, m) * (m, 1) + (n, 1)
 */
vector_2d Dense::get_output(const vector_2d &input) {
	if(m_weights.size() != input.size())
		throw std::invalid_argument("Input size is not proper");

	vector_2d new_weights(m_bias[0].size(), vector_1d(1));
	for (size_t i = 0; i < m_weights[0].size(); ++i) {
		dotprod(m_weights, i, input, m_bias[0][i], m_activation_type, new_weights[i][0]);
	}
	return activaction(new_weights);
}

// Flatten ========================================================================================
// public ---------------------------------------------------------------------------------------

/*
 * Creates Resize input into column vector.
 * input 	(n ,  m)
 * output	(n*m, 1)
 */
vector_2d Flatten::get_output(const vector_2d &input) {
	int orig_row = input.size();
	int orig_col = input[0].size();
	vector_2d flat(orig_row * orig_col, vector_1d(1));

	for (int i = 0; i < orig_row; ++i) {
		for (int j = 0; j < orig_col; ++j) {
			int offset = i * orig_col + j;
			flat[offset][0] = input[i][j];
		}
	}
	return flat;
}

// NeuralNetwork ==================================================================================

void NeuralNetwork::load_weights(const std::string &input_fname) {
	std::ifstream fin(input_fname.c_str());
	std::string layer_type = "";
	std::string tmp_str = "";
	int tmp_int = 0;
	int m_layers_cnt;

	fin >> tmp_str >> m_layers_cnt;
	for (int layer = 0; layer < m_layers_cnt; ++layer) {
		fin >> tmp_str >> tmp_int >> layer_type;

		NnLayer *l = 0;
		if (layer_type == "Flatten") {
			l = new Flatten();
		} else if (layer_type == "Dense") {
			l = new Dense();
		}
		if (l == 0) {
			cout << "Layer is empty," << endl;
			return;
		}
		l->load_weights(fin);
		add_layer(l);
	}
	fin.close();
}

/*
 *	Add new layer to NeuralNetwork class.
 *	Implemented layer types:
 *		Flatten
 *		Dense
 */
void NeuralNetwork::add_layer(NnLayer *layer) {
	m_layers.push_back(layer);
}

/*
 * Calculate the prediction for the given input
 * get_output input and output is vector_2d for generalization
 */
vector_2d NeuralNetwork::predict(const vector_2d &input) {
	vector_2d temp = input;
	for(auto layer : m_layers) {
		temp = layer->get_output(temp);
	}
	return temp;
}

int NeuralNetwork::classify(const vector_2d &input) {
	vector_2d result  = predict(input);
	return std::max_element(result.begin(), result.end()) - result.begin();
}

/*
 * Free the memory of the given layers.
 */
NeuralNetwork::~NeuralNetwork() {
	for(auto layer : m_layers) {
		delete layer;
	}
}

