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
#include <vector>

// TODO using vector_1d = std::vector<float>; - modernebb
// TODO Double <-> float
typedef std::vector<float>     	 vector_1d;
typedef std::vector< vector_1d > vector_2d;
typedef std::vector< vector_2d > vector_3d;

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
	virtual vector_2d get_output(const vector_2d &input) = 0;
	virtual ~NnLayer() {}
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
//	void dotprodWithRelu(int col, const vector_2d &input, float &output);

public:
	Dense(const vector_2d &weights, const  vector_2d &bias, const std::string &a_type);
	virtual vector_2d get_output(const vector_2d &input);
	virtual ~Dense();
};

// NnLayer ========================================================================================
class NeuralNetwork {
private:
	std::vector<NnLayer*> m_layers;

public:
	NeuralNetwork() {}
	virtual ~NeuralNetwork();

	void add_layer(NnLayer *layer);
	vector_2d predict(const vector_2d &input);
};

#endif /* NNLAYER_H_ */
