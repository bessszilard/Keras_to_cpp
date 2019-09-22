/*
 * NnLayer.h
 *
 *  Created on: Sep 20, 2019
 *      Author: szilard
 */

#ifndef NNLAYER_H_
#define NNLAYER_H_

#include <string>
#include <vector>

typedef std::vector<float>     float_vector; // 1D
typedef std::vector<std::vector<float> > vector_2d;

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

class Dense   : public NnLayer {
private:
	vector_2d m_weights;
	vector_2d m_bias;
	std::string m_activation_type;

private:
	vector_2d softmax(const vector_2d &input);

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
