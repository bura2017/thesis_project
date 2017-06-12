#include "DualSimplex.h"
#include "CublasMultip.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix0, dev_matrix1, right_temp;
static double *dev_row;
static int flag;

static void memInit (Matrix &matrix) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix0.rows = matrix.rows;
  dev_matrix0.cols = matrix.cols;
  dev_matrix0.m = matrix.m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix0.e, sizeof (double) * dev_matrix0.m * dev_matrix0.cols));
  CHECK_CUDA(cudaMemcpy(dev_matrix0.e, matrix.e, sizeof(double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  dev_matrix1.rows = matrix.rows;
  dev_matrix1.cols = matrix.cols;
  dev_matrix1.m = matrix.m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix1.e, sizeof (double) * dev_matrix1.m * dev_matrix1.cols));

  right_temp.rows = matrix.cols;
  right_temp.cols = matrix.cols;
  int x = (right_temp.cols - 1) / BLOCK_SIZE + 1;
  x *= BLOCK_SIZE;
  right_temp.m = x;
  CHECK_CUDA (cudaMalloc ((void**)&right_temp.e, sizeof(double) * x * x));
  iden_matr<<<x, x>>>(right_temp);

  CHECK_CUDA (cudaMalloc ((void**)&dev_row, sizeof (double) * matrix.cols));

}
static void memFree () {
  cudaFree (dev_matrix0.e);
  cudaFree (dev_matrix1.e);
  cudaFree(right_temp.e);
  cudaFree (dev_row);
}
static int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    if (cmp(matrix.e[i + 0 * matrix.m],0) == -1) {
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
          double val1 = matrix.e[i + ret * matrix.m] * matrix.e[row + j * matrix.m];
          double val2 = matrix.e[i + j * matrix.m] * matrix.e[row + ret * matrix.m];
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
static void fill_right_trans(d_matrix matrix, int col, double *row) {
  int j = threadIdx.x;
  __shared__ double piv_num;
  if (j == col) {
    piv_num = matrix.e[col + col * matrix.m];
  }
  __syncthreads();

  if (j != col) {
    int n = col + j * matrix.m;
    matrix.e[n] = piv_num * row[j];
  }
}
int gpuDualSimplexCublas (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  memInit(matrix);

  double piv_box[2];
  double piv_row[matrix.cols];

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
    //std::cout << flag << ' ' << pivot_row << ' ' << pivot_column << std::endl;

    piv_box[0] = - 1 / matrix.e[pivot_row + pivot_col * matrix.m];
    for (int i = 0; i < matrix.cols; i++) {
      piv_row[i] = matrix.e[pivot_row + i * matrix.m];
    }

    int side = right_temp.m;
    iden_matr<<<side,side>>> (right_temp);
    CHECK_CUDA(cudaMemcpy(&(right_temp.e[pivot_col + pivot_col * right_temp.m]), &(piv_box[0]),
        sizeof(double) * 1, cudaMemcpyHostToDevice));
    side = right_temp.cols;
    CHECK_CUDA (cudaMemcpy (dev_row, piv_row, sizeof (double) * side, cudaMemcpyHostToDevice));
    fill_right_trans <<<1,side>>>(right_temp, pivot_col, dev_row);

    if (flag % 2) {
      cublas_multip(dev_matrix0, right_temp, dev_matrix1);
      CHECK_CUDA(cudaMemcpy(matrix.e, dev_matrix1.e,sizeof(double) * matrix.m * matrix.cols,
          cudaMemcpyDeviceToHost));
    } else {
      cublas_multip(dev_matrix1, right_temp, dev_matrix0);
      CHECK_CUDA(cudaMemcpy(matrix.e, dev_matrix0.e,sizeof(double) * matrix.m * matrix.cols,
          cudaMemcpyDeviceToHost));
    }
  }
}
