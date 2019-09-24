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
//#include <thread>

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
Dense::Dense(const std::vector<std::vector<nn_cal_type> > &weights,
				 const std::vector<std::vector<nn_cal_type> > &bias,
				 const std::string &a_type) :
	m_weights(weights), m_bias(bias), m_activation_type(a_type)
{
	if(1 < m_bias.size())
		throw std::invalid_argument("Invalid bias size");
}

Dense::~Dense() {}

inline void dotprod(vector_2d &m_weights, int col, const vector_2d &input, nn_cal_type m_bias, std::string act_type, nn_cal_type &output) {
	nn_cal_type sum = 0.0f;
	for (size_t k = 0; k < m_weights.size(); ++k) {
		sum += m_weights[k][col] * input[k][0];
	}
	sum += m_bias;
	output = sum;
}
/*
vector_2d Dense::get_output(const vector_2d &input) {
	// TOOD check dimensions
	vector_2d new_weights(m_bias[0].size(), vector_1d(1));
//	std::thread threads[m_weights[0].size()];

	// Y = transp(W) * X + transp(B)
	for (size_t i = 0; i < m_weights[0].size(); ++i) {
		dotprodWithRelu(m_weights, i, input, m_bias[0][i], m_activation_type, new_weights[i][0]);
//		threads[i] = std::thread(dotprodWithRelu, std::ref(m_weights), i, std::ref(input), m_bias[0][i], m_activation_type, std::ref(new_weights[i][0]));
	}
//	for (size_t i = 0; i < m_weights[0].size(); ++i) {
//		threads[i].join();
//	}
	return activaction(new_weights)s;
}
*/

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

/*
 * Free the memory of the given layers.
 */
NeuralNetwork::~NeuralNetwork() {
//	for(size_t i = 0; i < m_layers.size(); ++i) {
	for(auto layer : m_layers) {
		delete layer;
	}
}

