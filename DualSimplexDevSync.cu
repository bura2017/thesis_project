#include "DualSimplex.h"
#include "Epsilon.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix;
static int *num_cols, *dev_num_cols;
static int *dev_piv_col;
static double *dev_col;
static int flag;

static void memInit (Matrix &matrix, int &blocks) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix.rows = matrix.rows;
  dev_matrix.cols = matrix.cols;
  dev_matrix.m = matrix.m;
  blocks = (matrix.cols - 2) / BLOCK_SIZE + 1;
  if (matrix.m < BLOCK_SIZE * blocks) {
    ERROR ("SyncDev dual simplex needs bigger matrix");
  }
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof (double) * matrix.m * (blocks * BLOCK_SIZE + 1)));
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof(double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  int temp = (blocks - 1) / BLOCK_SIZE + 1;
  temp *= BLOCK_SIZE;
  num_cols = new int[temp];
  CHECK_NULL(num_cols);
  CHECK_CUDA (cudaMalloc ((void**)&dev_num_cols, sizeof(int) * temp));

  CHECK_CUDA (cudaMalloc((void**)&dev_col, sizeof(double) * matrix.rows));
  CHECK_CUDA (cudaMalloc ((void**)&dev_piv_col, sizeof (int) * 1));
}
static void memFree () {
  cudaFree (dev_matrix.e);
  cudaFree (dev_col);

  delete [] num_cols;
  cudaFree (dev_num_cols);
  cudaFree (dev_piv_col);

}
int gpuDualSimplexSyncDev (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  int pivot_row = pivotRow (matrix);
  if (!pivot_row) {
    memFree();
    return flag;
  }

  int pivot_col = pivotColumn (matrix, pivot_row);
  if (!pivot_col) {
    memFree();
    return -flag;
  }

  int blocks;
  memInit(matrix, blocks);

  while (1) {
    flag ++;
    if (flag > 1000000000000) {
      std::cout << "ups" << std::endl;
      memFree();
      return 0;
    }
    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    int temp = (blocks - 1) / BLOCK_SIZE + 1;
    temp *= BLOCK_SIZE;
    for (int i = 0; i < temp; i++) {
      num_cols[i] = 0;
    }
    CHECK_CUDA (cudaMemcpy (dev_num_cols, num_cols, sizeof(int) * temp, cudaMemcpyHostToDevice));

    double div = - matrix.e[pivot_row + pivot_col * matrix.m];
    for (int i = 0; i < matrix. rows; i++) {
      matrix.e[i + pivot_col * matrix.m] /= div;
    }
    CHECK_CUDA (cudaMemcpy (dev_col, &(matrix.e[0 + pivot_col * dev_matrix.m]),
        sizeof (double) * matrix.rows, cudaMemcpyHostToDevice));

    double piv_row_el = matrix.e[pivot_row];
    for (int i = 0; i < matrix.rows; i++) {
      matrix.e[i] += matrix.e[i + pivot_col * matrix.m] * piv_row_el;
    }

    int pivot_row_next = pivotRow(matrix);
    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
    matrixTransformDev<<<blocks, threads>>>(dev_matrix, pivot_row, pivot_col, dev_col,
        pivot_row_next, dev_num_cols);
    /*int temp_bl = (blocks - 1) / BLOCK_SIZE + 1;
    matrixChoosePivot<<<temp_bl, threads>>> (dev_matrix, dev_num_cols, dev_num_col);
    CHECK_CUDA (cudaMemcpy (&pivot_col, dev_piv_col, sizeof (int) * 1, cudaMemcpyDeviceToHost));
    /**/
    /*if (flag == 2) {
      CHECK_CUDA (cudaMemcpy (&(matrix.e[matrix.m]), &(dev_matrix.e[dev_matrix.m]),
          sizeof (double) * matrix.m * (matrix.cols - 1), cudaMemcpyDeviceToHost));
      matrix.print("Check.txt");
    }/**/
    CHECK_CUDA (cudaMemcpy (num_cols, dev_num_cols, sizeof (int) * blocks, cudaMemcpyDeviceToHost));
    /*cudaDeviceSynchronize();
    for (int i = 0; i < blocks; i++) {
      std::cout << num_cols[i] << ' ';
    }
    std::cout << std::endl;
    /**/
    pivot_col = 0;
    pivot_row = pivot_row_next;
    if (!pivot_row) {
      memFree();
      return flag + 1;
    }

    for (int j = 0; j < blocks; j++) {
      if (num_cols[j] != 0) {
        CHECK_CUDA (cudaMemcpy (&(matrix.e[0 + num_cols[j] * matrix.m]),
            &(dev_matrix.e[0 + num_cols[j] * dev_matrix.m]),
            sizeof (double) * matrix.rows, cudaMemcpyDeviceToHost));
        if (pivot_col == 0) {
          pivot_col = num_cols[j];
        } else {
          for (int i = 0; i < matrix.rows; i++) {
            double val1 = - matrix.e[i + pivot_col * matrix.m] / matrix.e[pivot_row + pivot_col * matrix.m];
            double val2 = - matrix.e[i + num_cols[j] * matrix.m] / matrix.e[pivot_row + num_cols[j] * matrix.m];
            int c = cmp(val1,val2);
            if (c == 1) {
              pivot_col = num_cols[j];
              break;
            }
            if (c == -1) {
              break;
            }
          }
        }
      }
    }
    if (!pivot_col) {
      memFree();
      return -flag - 1;
    }
  }
}
