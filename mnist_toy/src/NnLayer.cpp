/*
 * NnLayer.cpp
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#include <math.h>
#include "NnLayer.h"

// Dense -----------------------------------------------------------------------------------------
Flatten::Flatten(int x) {
	std::cout << x << " Flatten constructor" << std::endl;
}

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
// private ---------------------------------------------------------------------------------------
Matrix<float> Dense::activation(const Matrix<float> &input) {
	if(input.getWidth() != 1)
		throw std::invalid_argument("Activation couldn't completed, because this is not a column vector");
	int orig_row = input.getHeight();
	Matrix<float> result(orig_row, 1);

	if(m_activation_type == "relu") {
		for(int i = 0; i < orig_row; ++i) {
			float temp = input.get(i, 0);
			if (temp < 0)
				temp = 0;
			result.put(i, 0, temp);
		}
		return result;
	}
	else if(m_activation_type == "softmax") {
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

Dense::~Dense() {

}

NnLayer::~NnLayer() {

}

Matrix<float> Dense::get_output(const Matrix<float> &input) {
	// TOOD check dimensions
	return activation(m_weights.transpose() * input + m_bias.transpose());
}

