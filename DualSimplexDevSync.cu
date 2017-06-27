#include "DualSimplex.h"
#include "Epsilon.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix;
static int *num_cols, *dev_num_cols;
static double *dev_col;
static int flag;

static void memInit (Matrix &matrix, int &blocks) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  dev_matrix.rows = matrix.rows;
  dev_matrix.cols = matrix.cols;
  dev_matrix.m = matrix.m;
  blocks = (matrix.cols - CPU_COLS - 1) / BLOCK_SIZE + 1;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof (double) * matrix.m * (blocks * BLOCK_SIZE + CPU_COLS)));
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof(double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  int sup = blocks % (BLOCK_SIZE * 2 - 1);
  sup = (sup > 0 ? BLOCK_SIZE * 2 - sup : 1);
  num_cols = new int[blocks + sup];
  CHECK_NULL(num_cols);
  for (int i = 0; i < blocks + sup; i++) {
    num_cols[i] = 0;
  }
  CHECK_CUDA (cudaMalloc ((void**)&dev_num_cols, sizeof(int) * (blocks + sup)));
  CHECK_CUDA (cudaMemcpy (dev_num_cols, num_cols, sizeof (int) * (blocks + sup), cudaMemcpyHostToDevice));

  CHECK_CUDA (cudaMalloc((void**)&dev_col, sizeof(double) * matrix.rows));
}
static void memFree () {
  cudaFree (dev_matrix.e);
  cudaFree (dev_col);

  delete [] num_cols;
  cudaFree (dev_num_cols);

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

    for (int i = 0; i < blocks; i++) {
      num_cols[i] = 0;
    }
    CHECK_CUDA (cudaMemcpy (dev_num_cols, num_cols, sizeof(int) * blocks, cudaMemcpyHostToDevice));

    double div = - matrix.e[pivot_row + pivot_col * matrix.m];
    for (int i = 0; i < matrix. rows; i++) {
      matrix.e[i + pivot_col * matrix.m] /= div;
    }
    CHECK_CUDA (cudaMemcpy (dev_col, &(matrix.e[0 + pivot_col * matrix.m]),
        sizeof (double) * matrix.rows, cudaMemcpyHostToDevice));

    double piv_row_el = matrix.e[pivot_row];
    for (int i = 0; i < matrix.rows; i++) {
      matrix.e[i] += matrix.e[i + pivot_col * matrix.m] * piv_row_el;
    }

    //===========TRANSFORM ON DEVICE==================================================
    int pivot_row_next = pivotRow(matrix);
    {
      dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
      matrixTransformDev<<<blocks, threads>>>(dev_matrix, pivot_row, pivot_col, dev_col,
          pivot_row_next, dev_num_cols);
    }
    //=================TRANSFORM ON CPU=====================================================
    /*for (int j = 1; j < CPU_COLS; j++) {
      if (j != pivot_col) {
        double piv_row_el = matrix.e[pivot_row + j * matrix.m];
        for (int i = 0; i < matrix.rows; i++) {
          matrix.e[i + j * matrix.m] += matrix.e[i + pivot_col * matrix.m] * piv_row_el;
        }
      }
    }/**/

    pivot_col = 0;
    pivot_row = pivot_row_next;
    if (!pivot_row) {
      memFree();
      return flag + 1;
    }

    //==================CHOOSE MIN COL ON DEVICE===============================================
    {
      dim3 threads(BLOCK_SIZE * 2,BLOCK_SIZE * 2);
      int bound = (blocks - 1) / (BLOCK_SIZE * 2) + 1;
      matrixChoosePivot<<<1, threads>>> (dev_matrix, dev_num_cols, pivot_row, bound);
    }/**/
    //================CHOOSE MIN ON CPU=============================================
    /*int piv_col_cpu = 0;
    {
      int temp_cols = matrix.cols;
      matrix.cols = CPU_COLS;
      piv_col_cpu = pivotColumn(matrix, pivot_row);
      matrix.cols = temp_cols;
    }/**/

    //================SYNCHRONIZE===============
    CHECK_CUDA (cudaMemcpy (&pivot_col, dev_num_cols, sizeof (int) * 1, cudaMemcpyDeviceToHost));
    if (pivot_col != 0) {
      CHECK_CUDA (cudaMemcpy (&(matrix.e[0 + pivot_col * matrix.m]),
          &(dev_matrix.e[0 + pivot_col * dev_matrix.m]),
          sizeof (double) * matrix.rows, cudaMemcpyDeviceToHost));
    }

    /*if (pivot_col == 0) {
      pivot_col = piv_col_cpu;
    }/**/
    if (!pivot_col) {
      memFree();
      return -flag - 1;
    }
    /*if ((pivot_col != piv_col_cpu) && (piv_col_cpu != 0)) {
      for (int i = 0; i < matrix.rows; i++) {
        double val1 = - matrix.e[i + pivot_col * matrix.m] / matrix.e[pivot_row + pivot_col * matrix.m];
        double val2 = - matrix.e[i + piv_col_cpu * matrix.m] / matrix.e[pivot_row + piv_col_cpu * matrix.m];
        int c = cmp(val1,val2);
        if (c == 1) {
          pivot_col = piv_col_cpu;
          break;
        }
        if (c == -1) {
          break;
        }
      }
    }/**/
  }
}
