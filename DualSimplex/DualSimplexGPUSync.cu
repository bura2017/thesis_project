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

#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix;
static double *dev_col;
static int flag;

static void memInit (Matrix &matrix) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix.rows = matrix.rows;
  dev_matrix.cols = matrix.cols;
  dev_matrix.m = matrix.m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof (double) * matrix.m * matrix.cols));
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof(double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  CHECK_CUDA (cudaMalloc((void**)&dev_col, sizeof(double) * matrix.rows));
}
static void memFree () {
  cudaFree (dev_matrix.e);
  cudaFree (dev_col);

}

int gpuDualSimplexSync (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  memInit(matrix);

  while (1) {
    flag ++;

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      memFree();
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      memFree();
      return 0;
    }

    matrixTransformSync (matrix, pivot_row, pivot_col, dev_matrix, dev_col);
  }
}
