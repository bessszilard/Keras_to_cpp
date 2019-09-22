/*
 * NnLayer.cpp
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#include <math.h>
#include "NnLayer.h"
#include <time.h>
#include <stdexcept>

// Dense ==========================================================================================
// private ----------------------------------------------------------------------------------------
vector_2d Dense::softmax(const vector_2d &input) {
	if(input[0].size() != 1)
		throw std::invalid_argument("Activation couldn't completed, because this is not a column vector");
	int orig_row = input.size();
	vector_2d result(orig_row, float_vector(1));
	if(m_activation_type == "softmax") {
		float sum = 0.0;
		for(int k = 0; k < orig_row; ++k) {
		  float temp = exp(input[k][0]);
		  result[k][0] = temp;
		  sum += temp;
		}
		for(int k = 0; k < orig_row; k++)
			result[k][0] /= sum;
		return result;
	}
	else {
		throw std::invalid_argument("Unknown activation function");
	}
}

// public ---------------------------------------------------------------------------------------
//NnLayer::NnLayer() {
Dense::Dense(const std::vector<std::vector<float> > &weights,
				 const std::vector<std::vector<float> > &bias,
				 const std::string &a_type) :
	m_weights(weights), m_bias(bias), m_activation_type(a_type)
{
	//TODO throw exception if bias width is larger than 0
}
Dense::~Dense() {}

vector_2d Dense::get_output(const vector_2d &input) {
	// TOOD check dimensions
	vector_2d new_weights(m_bias[0].size(), float_vector(1));
	for (size_t i = 0; i < m_weights[0].size(); ++i) {
		float sum = 0.0f;
		for (size_t k = 0; k < m_weights.size(); ++k) {
			sum += m_weights[k][i] * input[k][0];
		}
		sum += m_bias[0][i];
		if (m_activation_type == "relu") {
			if (sum < 0)
				sum = 0;
		}
		new_weights[i][0] = sum + m_bias[0][i];
	}
	if (m_activation_type == "relu")
		return new_weights;
	return softmax(new_weights);
}

// Flatten ========================================================================================
// public ---------------------------------------------------------------------------------------
vector_2d Flatten::get_output(const vector_2d &input) {
	int orig_row = input.size();
	int orig_col = input[0].size();
	vector_2d flat(orig_row * orig_col, float_vector(1));

	for (int i = 0; i < orig_row; ++i) {
		for (int j = 0; j < orig_col; ++j) {
			int offset = (i * orig_col + j);
			flat[offset][0] = input[i][j];
		}
	}
	return flat;
}

// NeuralNetwork ==================================================================================
void NeuralNetwork::add_layer(NnLayer *layer) {
	m_layers.push_back(layer);
}

vector_2d NeuralNetwork::predict(const vector_2d &input) {
	vector_2d temp = input;
	for(auto layer : m_layers) {
		temp = layer->get_output(temp);
	}
	return temp;
}

NeuralNetwork::~NeuralNetwork() {
	for(int i = 0; i < (int)m_layers.size(); ++i) {
		delete m_layers[i];
	}
}

