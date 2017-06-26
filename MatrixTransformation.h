/*
 * MatrixTransformation.h
 *
 *  Created on: 13 июня 2017 г.
 *      Author: valerius
 */

#ifndef MATRIXTRANSFORMATION_H_
#define MATRIXTRANSFORMATION_H_

#include "Matrix.h"

struct data_full_task {
  Matrix *matrix;
  int piv_col;
  int piv_row;
  double *dev_col;
  Matrix *pin_matrix;
  d_matrix dev_matrix;
  cudaStream_t stream;

};
struct data_async {
  double *dev_col;
  Matrix *pin_matrix;
  d_matrix dev_matrix;
  cudaStream_t stream;

  data_async &operator=(data_full_task &data) {
    dev_col = data.dev_col;
    pin_matrix = data.pin_matrix;
    dev_matrix = data.dev_matrix;
    stream = data.stream;

    return *this;
  }
};
int matrixTransformCpu(Matrix &matrix, const int row, const int col);
int matrixTransformAsync(Matrix &matrix, const int row, const int col, data_async &data0, data_async &data1);
int matrixTransformSync(Matrix &matrix, const int row, const int col, d_matrix &dev_matrix, double *dev_col);

__global__ void matrixTransform(d_matrix matrix, int piv_row, double *col_e);
__global__ void matrixTransformSync(d_matrix matrix, int piv_row, int piv_col, double *col_e);
int matrixTransformDouble (data_full_task data0, data_full_task data1);

#endif /* MATRIXTRANSFORMATION_H_ */
