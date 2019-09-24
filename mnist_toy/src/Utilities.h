/*
 * Utilities.h
 *
 *  Created on: Sep 23, 2019
 *      Author: szilard
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <fstream>
#include "NnLayer.h"

using std::ifstream;

struct clock_bounds {
	clock_t start;
	clock_t end;
};

struct Utilities {
	static vector_2d read_from_file(const std::string &fname);
	static vector_2d read_from_binary_file(const std::string &fname);
};

class Clocks {
private:
	float m_tExecution;
	float m_tInit;
	float m_tFileRead;
	float m_tPrediction;
	int   m_iterations;
	clock_bounds m_clkExecuition, m_clkInit, m_clkFileRead, m_clkPrediction;

	static double getms(clock_bounds clk);
	static int getpercent(clock_bounds clk, clock_bounds overall);

public:
	Clocks () {
		m_tExecution    = 0.0f;
		m_tInit         = 0.0f;
		m_tFileRead     = 0.0f;
		m_tPrediction   = 0.0f;
	    m_iterations 	= 0;
	}

	void start_Init();
	void start_FileRead();
	void start_Prediction();
	void iteration_finished();

	float get_Init_average()		{ return m_tInit 		/ m_iterations; }
	float get_FileRead_average()	{ return m_tFileRead 	/ m_iterations; }
	float get_Prediction_average()  { return m_tPrediction  / m_iterations; }
	float get_Execuition_average()  { return m_tExecution   / m_iterations; }

	float get_Init_percent()		{ return (int)(m_tInit 		 / m_tExecution * 100); }
	float get_Float_percent()       { return (int)(m_tFileRead 	 / m_tExecution * 100); }
	float get_Prediction_percent()  { return (int)(m_tPrediction / m_tExecution * 100); }
};

#endif /* UTILITIES_H_ */
