#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix;
static float *dev_col;
static int flag;

static void memInit (const int rows, const int cols, const int m) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix.rows = rows;
  dev_matrix.cols = cols;
  dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof (float) * m * cols));

  CHECK_CUDA (cudaMalloc((void**)&dev_col, sizeof(float) * rows));
}
static void memFree () {
  cudaFree (dev_matrix.e);
  cudaFree (dev_col);

}

int gpuDualSimplexSync (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  memInit(matrix.rows, matrix.cols, matrix.m);

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
