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
    if (flag % 1000000 == 0) {
      memFree();
      return 0;
    }

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
    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    matrixTransformSync (matrix, pivot_row, pivot_col, dev_matrix, dev_col);
  }
}
