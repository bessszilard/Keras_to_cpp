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

public:
	NnLayer() {};
//	NnLayer(const std::vector<std::vector<float> > &weights, const  std::vector<std::vector<float> > &bias, const std::string &a_type);
	virtual Matrix<float> get_output(const Matrix<float> &input) = 0;
	virtual ~NnLayer();
	int get_inputSize()  { return m_input_size;  }
	int get_outputSize() { return m_output_size; }

	void set_inputSize(int size)  { m_input_size = size;  }
	void set_outputSize(int size) { m_output_size = size; }
};

class Flatten : public NnLayer {
public:
	Flatten(int);
	virtual ~Flatten() {};
	Matrix<float> get_output(const Matrix<float> &input);
};

class Dense   : public NnLayer {
private:
//	int m_input_size;
//	int m_output_size;
	Matrix<float> m_weights;
	Matrix<float> m_bias;
	std::string m_activation_type;

private:
	Matrix<float> activation(const Matrix<float> &input);

public:
	Dense(const std::vector<std::vector<float> > &weights, const  std::vector<std::vector<float> > &bias, const std::string &a_type);
	virtual Matrix<float> get_output(const Matrix<float> &input);
	virtual ~Dense();
//	int get_inputSize()  { return m_input_size;  }
//	int get_outputSize() { return m_output_size; }

};

#endif /* NNLAYER_H_ */
