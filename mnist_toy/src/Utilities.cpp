/*
 * Utilities.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: szilard
 */
#include "Utilities.h"
#include <iostream>

vector_2d Utilities::read_from_file(const std::string &fname) {
	int depth, rows, cols;
	ifstream fin(fname.c_str());
	if(fin.fail())
		throw std::invalid_argument( "can't open " + fname);
	fin >> depth >> rows >> cols;

	vector_3d data = vector_3d(depth, vector_2d(rows, vector_1d(cols)));
	for (int d = 0; d < depth; ++d) {
		for (int r = 0; r < rows; ++r) {
			char tmp_char;
			fin >> tmp_char;
			for (int n = 0; n < cols; ++n) {
				fin >> data[d][r][n];
			}
			fin >> tmp_char;
		}
	}

	fin.close();
	return data[0];
}

vector_2d Utilities::read_from_binary_file(const std::string &fname) {
	int rows = 28, cols = 28;
	ifstream binFile(fname.c_str(), std::ios::binary | std::ios::in);

	if(binFile.fail())
		throw std::invalid_argument( "can't open " + fname);
	binFile.seekg(0, std::ios::end);
	int fileSize = (int)binFile.tellg();

	if(fileSize != rows * cols)
		throw std::invalid_argument( "Invalid file size " + fname);

	binFile.seekg(0, std::ios::beg);
	vector_2d data = vector_2d(rows, vector_1d(cols));
	unsigned char array[28][28];
	binFile.read((char*)&array, 28*28);						// copy the whole file content into 28x28 array

	for (int r = 0; r < rows; ++r) {
		for (int n = 0; n < cols; ++n) {
			data[r][n] = (nn_cal_type)(array[r][n]) / 256;	// normalization
		}
	}

	binFile.close();
	return data;
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
	//TODO combine start functions into one
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
