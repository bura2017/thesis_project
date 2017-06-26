#include "DualSimplex.h"
#include <iostream>

static int flag;
static data_full_task data0, data1;

static void memInit(Matrix &matrix0, Matrix &matrix1) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  int size = MAX_BLOCKS * matrix0.m;

  data0.matrix = &matrix0;
  data0.dev_matrix.rows = matrix0.rows;
  data0.dev_matrix.cols = MAX_BLOCKS;
  data0.dev_matrix.m = matrix0.m;
  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_matrix.e, sizeof(double) * size));

  data1.matrix = &matrix1;
  data1.dev_matrix.rows = matrix1.rows;
  data1.dev_matrix.cols = MAX_BLOCKS;
  data1.dev_matrix.m = matrix1.m;
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_matrix.e, sizeof(double) * size));

  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_col, sizeof(double) * matrix0.m));
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_col, sizeof(double) * matrix1.m));

  CHECK_CUDA(cudaStreamCreate(&data0.stream));
  CHECK_CUDA(cudaStreamCreate(&data1.stream));

  data0.pin_matrix = new Matrix(matrix0.rows, MAX_BLOCKS, cudaHostAllocDefault, matrix0.m - matrix0.rows);
  data1.pin_matrix = new Matrix(matrix1.rows, MAX_BLOCKS, cudaHostAllocDefault, matrix1.m - matrix1.rows);
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

int *gpuDualSimplexDouble (Matrix &matrix0, Matrix &matrix1) {
  CHECK_NULL(matrix0.e);
  CHECK_NULL(matrix1.e);

  //std::cout << "//=======================" << std::endl;
  memInit(matrix0, matrix1);
  int *check = new int[2];
  check[0] = 0;
  check[1] = 0;

  while (1) {
    flag ++;
    //std::cout << flag << std::endl;
    if (flag % 1000000000 == 0) {
      std::cout << "ups" << std::endl;
      memFree ();
      return 0;
    }

    if (check[0] == 0) {
      data0.piv_row = pivotRow (matrix0);
      if (data0.piv_row == 0) {
        check[0] = flag;
      }
    }

    if (check[1] == 0) {
      data1.piv_row = pivotRow (matrix1);
      if (data1.piv_row == 0) {
        check[1] = flag;
      }
    }

    if (check[0] == 0) {
      data0.piv_col = pivotColumn (matrix0, data0.piv_row);
      if (data0.piv_col == 0) {
        check[0] = -flag;
      }
    }

    if (check[1] == 0) {
      data1.piv_col = pivotColumn (matrix1, data1.piv_row);
      if (data1.piv_col == 0) {
        check[1] = -flag;
      }
    }

    //std::cout << flag << ' ' << data1.piv_row << ' ' << data1.piv_col << std::endl;

    if (check[0] == 0 && check[1] == 0) {
      matrixTransformDouble (data0, data1);
    }
    if (check[0] == 0 && check[1] != 0) {
      data_async data_0, data_1;
      data_0 = data0;
      data_1 = data1;
      matrixTransformAsync (*data0.matrix, data0.piv_row, data0.piv_col, data_0, data_1);
    }
    if (check[0] != 0 && check[1] == 0) {
      data_async data_0, data_1;
      data_0 = data0;
      data_1 = data1;
      matrixTransformAsync (*data1.matrix, data1.piv_row, data1.piv_col, data_0, data_1);
    }
    if (check[0] != 0 && check[1] != 0) {
      memFree();
      return check;
    }
  }
}
