/*
 * MatrixTransformation.h
 *
 *  Created on: 13 июня 2017 г.
 *      Author: valerius
 */

#ifndef MATRIXTRANSFORMATION_H_
#define MATRIXTRANSFORMATION_H_

#include "Matrix.h"

struct data_async {
  double *dev_col;
  Matrix *pin_matrix;
  d_matrix dev_matrix;
  cudaStream_t stream;
};

int matrixTransformCpu(Matrix &matrix, const int row, const int col);
int matrixTransformAsync(Matrix &matrix, const int row, const int col, const int threads, data_async &data0, data_async &data1);
int matrixTransformSync(Matrix &matrix, const int row, const int col, d_matrix &dev_matrix, double *dev_col);

__global__ void matrixTransform(d_matrix matrix, int piv_row, double *col_e);

#endif /* MATRIXTRANSFORMATION_H_ */
