/*
 * Utilities.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: szilard
 */
#include "Utilities.h"

vector_1d Utilities::read_1d_array(ifstream &fin, int cols) {
	vector_1d arr;
	arr.reserve(cols);
	float tmp_float;
	char tmp_char;
	fin >> tmp_char;
	for (int n = 0; n < cols; ++n) {
		fin >> tmp_float;
		arr.push_back(tmp_float);
	}
	fin >> tmp_char;
	return arr;
}

vector_2d Utilities::read_from_file(const std::string &fname) {
	int m_depth, m_rows, m_cols;
	ifstream fin(fname.c_str());
	if(fin.fail())
		throw std::invalid_argument( "can't open " + fname);
	fin >> m_depth >> m_rows >> m_cols;

	vector_3d data;
	data.reserve(m_depth * m_rows * m_cols);
	for (int d = 0; d < m_depth; ++d) {
		vector_2d tmp_single_depth;
		for (int r = 0; r < m_rows; ++r) {
			vector_1d tmp_row = read_1d_array(fin, m_cols);
			tmp_single_depth.push_back(tmp_row);
		}
		data.push_back(tmp_single_depth);
	}
	fin.close();
	return data[0];
}

double Clocks::getms(clock_bounds clk) {
	return ((double) (clk.end - clk.start) * 1000 / CLOCKS_PER_SEC);
}

int Clocks::getpercent(clock_bounds clk, clock_bounds overall) {
	return 100 * getms(clk) / getms(overall);
}

void Clocks::start_Init() {
	m_clkExecuition.start = clock();
	m_clkInit.start 	  = clock();
}
void Clocks::start_FileRead() {
	m_clkInit.end = clock();
	m_clkFileRead.start = m_clkInit.end;
}
void Clocks::start_Prediction() {
	m_clkFileRead.end = clock();
	m_clkPrediction.start = m_clkFileRead.end;
}
void Clocks::iteration_finished() {
	m_clkPrediction.end = clock();
	m_clkExecuition.end = m_clkPrediction.end   ;
	m_tInit         += getms(m_clkInit) 		;
	m_tFileRead     += getms(m_clkFileRead)	    ;
	m_tPrediction   += getms(m_clkPrediction)	;
	m_tExecution    += getms(m_clkExecuition)   ;

	m_iterations += 1;
}
