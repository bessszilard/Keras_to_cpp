/*
 * NnLayer.cpp
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#include <math.h>
#include "NnLayer.h"
#include <time.h>

// Dense ==========================================================================================
// private ----------------------------------------------------------------------------------------
Matrix<float> Dense::softmax(const Matrix<float> &input) {
	if(input.getWidth() != 1)
		throw std::invalid_argument("Activation couldn't completed, because this is not a column vector");
	int orig_row = input.getHeight();
	Matrix<float> result(orig_row, 1);
	if(m_activation_type == "softmax") {
		float sum = 0.0;
		for(int k = 0; k < orig_row; ++k) {
		  float temp = exp(input.get(k, 0));
		  result.put(k, 0, temp);
		  sum += temp;
		}
		result /= sum;
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

Matrix<float> Dense::get_output(const Matrix<float> &input) {
	// TOOD check dimensions
	Matrix<float> new_weights(m_bias.getWidth(), 1);
	for (int i = 0; i < m_weights.getWidth(); ++i) {
		float sum = 0.0f;
		for (int k = 0; k < m_weights.getHeight(); ++k) {
			sum += m_weights.get(k, i) * input.get(k, 0);
		}
		sum += m_bias.get(0, i);
		if (m_activation_type == "relu") {
			if (sum < 0)
				sum = 0;
		}
		new_weights.put(i, 0, sum + m_bias.get(0, i));
	}
	if (m_activation_type == "relu")
		return new_weights;
	return softmax(new_weights);
}

// Flatten ========================================================================================
// public ---------------------------------------------------------------------------------------
Matrix<float> Flatten::get_output(const Matrix<float> &input) {
	int orig_row = input.getHeight();
	int orig_col = input.getWidth();
	Matrix<float> flat(orig_row * orig_col, 1);

	for (int i = 0; i < orig_row; ++i) {
		for (int j = 0; j < orig_col; ++j) {
			int offset = (i * orig_col + j);
			flat.put(offset, 0, input.get(i, j));
		}
	}
	return flat;
}

// NeuralNetwork ==================================================================================
void NeuralNetwork::add_layer(NnLayer *layer) {
	m_layers.push_back(layer);
}

Matrix<float> NeuralNetwork::predict(const Matrix<float> &input) {
	Matrix<float> temp = input;
	for(auto layer : m_layers) {
//		clock_t start = clock();
		temp = layer->get_output(temp);
//		std::cout << x++ << " " << ((double) (clock() - start) * 1000 / CLOCKS_PER_SEC) << std:: endl;
	}
	return temp;
}

NeuralNetwork::~NeuralNetwork() {
	for(int i = 0; i < (int)m_layers.size(); ++i) {
		delete m_layers[i];
	}
}

