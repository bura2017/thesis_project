#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix;
static int flag;

static void memInit (const int rows, const int cols, const int m) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix.rows = rows;
  dev_matrix.cols = cols;
  dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof (double) * m * cols));

  //CHECK_CUDA (cudaMalloc((void**)&dev_col, sizeof(double) * rows));
}
static void memFree () {
  cudaFree (dev_matrix.e);
  //cudaFree (dev_col);

}
static int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    double val = matrix.e[i + 0 * matrix.m];
    if (cmp(val, 0) == -1) {
      return i;
    }
  }
  return 0;
}
static int pivotColumn(Matrix const &matrix, const int row) {
  int ret = 0;

  for (int j = 1; j < matrix.cols; j++) {
    if (cmp(matrix.e[row + j * matrix.m],0) == -1) {
      if (ret == 0) {
        ret = j;
      } else {
        for (int i = 0; i < matrix.rows; i++) {
          double val1 = - matrix.e[i + j * matrix.m] * matrix.e[row + ret * matrix.m];
          double val2 = - matrix.e[i + ret * matrix.m] * matrix.e[row + j * matrix.m];
          int c = cmp(val1,val2);
          if (c == -1) {
            ret = j;
            break;
          }
          if (c == 1) {
            break;
          }
        }
      }
    }
  }
  return ret;
}
__global__
static void simplexMatrix(d_matrix matrix, int piv_row, int piv_col) {
  __shared__ double cache[1024];
  __shared__ double matr[1024];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  //one column for one block

  cache[row] = matrix.e[row + piv_col * matrix.m];
  matr[row] = matrix.e[row + col * matrix.m];
  if (row == piv_row) {
    piv_row_el = matr[row];
  }
  __syncthreads();

  if (col != piv_col) {
    matr[row] += piv_row_el * cache[row];
  }
  __syncthreads();
  matrix.e[row + col * matrix.m] = matr[row];
}/**/
static void dualSimplex(Matrix &matrix, const int row, const int col) {
  double div = - matrix.e[row + col * matrix.m];
  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof (double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  simplexMatrix<<<matrix.cols, matrix.rows>>>(dev_matrix, row, col);

  CHECK_CUDA (cudaMemcpy (matrix.e, dev_matrix.e, sizeof (double) * matrix.m * matrix.cols,
      cudaMemcpyDeviceToHost));

  for (int j = 0; j < matrix.cols; j++) {
    matrix.e[row + j * matrix.m] = (j == col ? -1.0 : 0.0);
  }
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

    dualSimplex (matrix, pivot_row, pivot_col);
  }
}
