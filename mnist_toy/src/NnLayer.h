/*
 * NnLayer.h
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

//#pragma once
#ifndef NNLAYER_H_
#define NNLAYER_H_

#include <string>
#include "nnVector.h"

// NnLayer ========================================================================================
class NnLayer {
private:
	int m_input_size;
	int m_output_size;

	vector_2d m_weights;
	vector_2d m_bias;
	std::string m_activation_type;

public:
	NnLayer() {};
	virtual ~NnLayer() {}

	virtual void load_weights(std::ifstream &fin) = 0;
	virtual vector_2d get_output(const vector_2d &input) = 0;

	int get_inputSize()  { return m_input_size;  }
	int get_outputSize() { return m_output_size; }

	void set_inputSize(int size)  { m_input_size = size;  }
	void set_outputSize(int size) { m_output_size = size; }
};

// Flatten ========================================================================================
class Flatten : public NnLayer {
public:
	Flatten() {};
	virtual ~Flatten() {};
	void load_weights(std::ifstream &fin) {}
	vector_2d get_output(const vector_2d &input);
};

// Dense ==========================================================================================
class Dense   : public NnLayer {
private:
	vector_2d m_weights;
	vector_2d m_bias;
	std::string m_activation_type;

private:
	vector_2d activaction(const vector_2d &input);
	void dotprod(vector_2d &m_weights, int col, const vector_2d &input, nn_cal_type m_bias, std::string act_type, nn_cal_type &output);

public:
	Dense(const vector_2d &weights, const  vector_2d &bias, const std::string &a_type);
	Dense() {};
	virtual ~Dense();

	void load_weights(std::ifstream &fin);
	vector_2d get_output(const vector_2d &input);
};

// NnLayer ========================================================================================
class NeuralNetwork {
private:
	std::vector<NnLayer*> m_layers;

public:
	NeuralNetwork() {}
	virtual ~NeuralNetwork();

	void load_weights(const std::string &input_fname);
	void add_layer(NnLayer *layer);
	vector_2d predict(const vector_2d &input);
};

#endif /* NNLAYER_H_ */
