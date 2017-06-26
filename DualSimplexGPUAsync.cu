#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static int flag;
static data_async data0, data1;

static void memInit(const int rows, const int cols, int m) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  int size = MAX_BLOCKS * m;

  data0.dev_matrix.rows = rows;
  data0.dev_matrix.cols = MAX_BLOCKS;
  data0.dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_matrix.e, sizeof(double) * size));

  data1.dev_matrix.rows = rows;
  data1.dev_matrix.cols = MAX_BLOCKS;
  data1.dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_matrix.e, sizeof(double) * size));

  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_col, sizeof(double) * m));
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_col, sizeof(double) * m));

  CHECK_CUDA(cudaStreamCreate(&data0.stream));
  CHECK_CUDA(cudaStreamCreate(&data1.stream));

  data0.pin_matrix = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
  data1.pin_matrix = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
}
static void memFree () {
  cudaFree(data0.dev_matrix.e);
  cudaFree(data1.dev_matrix.e);

  cudaFree(data0.dev_col);
  cudaFree(data1.dev_col);

  CHECK_CUDA(cudaStreamDestroy(data0.stream));
  CHECK_CUDA(cudaStreamDestroy(data1.stream));

  data0.pin_matrix->freeHost();
  data1.pin_matrix->freeHost();
  delete data0.pin_matrix;
  delete data1.pin_matrix;
  //std::cout << flag << std::endl;
}

int gpuDualSimplexAsync (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  //std::cout << "===========simplex============" << std::endl;
  memInit(matrix.rows, matrix.cols, matrix.m);

  while (1) {
    flag ++;
    if (flag % 1000000000000 == 0) {
      std::cout << "ups" << std::endl;
      //memFree ();
      //return 0;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      memFree ();
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      memFree ();
      return -flag;
    }
//std::cout << flag << ' ' << pivot_row << ' ' << pivot_col << std::endl;

    matrixTransformAsync (matrix, pivot_row, pivot_col, data0, data1);
  }
}
