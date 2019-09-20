/*
 * NnLayer.h
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#ifndef NNLAYER_H_
#define NNLAYER_H_

#include "Matrix.h"

class NnLayer {
private:
	int m_input_size;
	int m_output_size;

	Matrix<float> m_weights;
	Matrix<float> m_bias;
	std::string m_activation_type;

private:
	Matrix<float> activation(const Matrix<float> &input);

public:
	NnLayer(const std::vector<std::vector<float> > &weights, const  std::vector<std::vector<float> > &bias, const std::string &a_type);
	virtual Matrix<float> get_output(const Matrix<float> &input);
	virtual ~NnLayer();
};

//class Flatten : public NnLayer {
//
//};
//
//class Dense   : public NnLayer {
//
//};

#endif /* NNLAYER_H_ */
