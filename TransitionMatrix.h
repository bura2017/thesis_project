/*
 * MatrixTransition.h
 *
 *  Created on: 13 июня 2017 г.
 *      Author: valerius
 */

#ifndef MATRIXTRANSITION_H_
#define MATRIXTRANSITION_H_

#include "Matrix.h"

int dev_trans_init(d_matrix &dev_trans, int side);
__global__
void fill_right_trans(d_matrix matrix, int col, double *row);
void modifyTransMatrAsync (int flag, int pivot_row, int pivot_col, d_matrix &temp_trans_1, d_matrix &temp_trans_2,
    d_matrix right_temp, cudaStream_t str_tr_ma);

#endif /* MATRIXTRANSITION_H_ */
