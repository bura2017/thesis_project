/**
 * Copyright (c) 2016 ISP RAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MATRIXTRANSFORMATION_H_
#define MATRIXTRANSFORMATION_H_

#include "../Matrix.h"

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
int matrixTransformAsync(Matrix &matrix, const int row, const int col,
    data_async &data0, data_async &data1);
int matrixTransformSync(Matrix &matrix, const int row, const int col,
    d_matrix &dev_matrix, double *dev_col);

__global__ void matrixTransform(d_matrix matrix, int piv_row, double *col_e);
__global__ void matrixTransformSync(d_matrix matrix, int piv_row, int piv_col, double *col_e);
int matrixTransformDouble (data_full_task data0, data_full_task data1);
__global__ void matrixTransformDev(d_matrix matrix, int piv_row, int piv_col, double *dev_col,
    int piv_row_next, int *num_cols);

#endif /* MATRIXTRANSFORMATION_H_ */
