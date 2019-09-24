/*
 * NnVector.h
 *
 *  Created on: Sep 24, 2019
 *      Author: szilard
 */

#ifndef NNVECTOR_H_
#define NNVECTOR_H_

#include <vector>

using nn_cal_type = float;
using vector_1d = std::vector<nn_cal_type>;
using vector_2d = std::vector< vector_1d >;
using vector_3d = std::vector< vector_2d >;

#endif /* NNVECTOR_H_ */
