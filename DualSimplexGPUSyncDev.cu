#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#define SHARED_MEM_SIZE 1024

static d_matrix dev_matrix;
static int *dev_piv_row, *dev_piv_col;
static int flag = 0;
static int *dev_er;
static void memInit (Matrix const &matrix) {
  flag = 0;
  dev_matrix.rows = matrix.rows;
  dev_matrix.cols = matrix.cols;
  dev_matrix.m = matrix.m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix.e, sizeof(double) * matrix.m * matrix.cols));
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof(double) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));

  CHECK_CUDA (cudaMalloc ((void**)&dev_piv_row, sizeof(int) * 1));
  CHECK_CUDA (cudaMalloc ((void**)&dev_piv_col, sizeof(int) * 1));

  CHECK_CUDA (cudaMalloc ((void**)&dev_er, sizeof(int)));
  int error = 0;
  CHECK_CUDA(cudaMemcpy(dev_er, &error, sizeof(int), cudaMemcpyHostToDevice));
}
static void memFree () {
  cudaFree(dev_matrix.e);

  cudaFree(dev_piv_row);
  cudaFree(dev_piv_col);

  cudaFree(dev_er);
}
__global__
void pivotRowColumn (d_matrix matrix, int *pivot_row, int *pivot_col) {
  __shared__ int cache[SHARED_MEM_SIZE + 1];
  __shared__ double col_e[SHARED_MEM_SIZE + 1];
  __shared__ double col_piv[SHARED_MEM_SIZE + 1];
  __shared__ int check;
  __shared__ int piv_row, piv_col;
  __shared__ double div_e, div_piv;
  register int row = threadIdx.x;

  cache[row] = 0;
  col_e[row] = matrix.e[row];
  if (threadIdx.x == 0) {
    *pivot_row = 0;
    *pivot_col = 0;
    check = 0;
    col_e[0] = 0.0;
    cache[matrix.rows] = 0;
  }
  __syncthreads();
  if (col_e[row] < -epsilon) {
    check = 1;
    cache[row] = row;
  }
  __syncthreads();
  if (check == 0) {
    return;
  } else {
    while (cache[0] == 0) {
      if (cache[row] == 0 && cache[row+1] != 0) {
        cache[row] = cache[row+1];
      }
    }
  }
  if(threadIdx.x == 0) {
    *pivot_row = cache[0];
    piv_row = cache[0];
    check = 0;
    piv_col = 0;
  }

  //==========================================================

  for (int k = 1; k < matrix.cols; k++) {
    __syncthreads();
    col_e[row] = matrix.e[row + k * matrix.m];
    __syncthreads();
    if (col_e[piv_row] < -epsilon) {
      if (row == piv_row) {
        div_e = - col_e[piv_row];
      }
      __syncthreads();
      if (piv_col == 0) {
        col_piv[row] = col_e[row];
        __syncthreads();
        if (threadIdx.x == 0) {
          piv_col = k;
          div_piv = div_e;
        }
      } else {
        cache[row] = cmp(col_e[row] * div_piv, col_piv[row] * div_e);
        if (cache[row] != 0) {
          check = 1;
        }
        __syncthreads();
        if (check == 0) {
          continue;
        }
        while (cache[0] == 0) {
          if (cache[row] == 0 && cache[row+1] != 0) {
            cache[row] = cache[row+1];
          }
        }
        if (cache[0] == -1) {
          col_piv[row] = col_e[row];
        }
        if(threadIdx.x == 0) {
          check = 0;
          if (cache[0] == -1) {
            piv_col = k;
            div_piv = div_e;
          }
        }
      }
    }
  }

  __syncthreads();
  if (piv_col == 0) {
    return;
  }
  //change pivot column
  col_piv[row] /= div_piv;
  matrix.e[row + piv_col * matrix.m] = col_piv[row];
  if (threadIdx.x == 0) {
    *pivot_col = piv_col;
  }
}

__global__
static void simplexMatrix(d_matrix matrix, int piv_row, int piv_col, int *err) {
  __shared__ double cache[1024];
  __shared__ double matr[1024];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  //one column for one block

  cache[row] = matrix.e[row + piv_col * matrix.m];
  matr[row] = matrix.e[row + col * matrix.m];
  if (row == piv_row) {
    *err = row;
    piv_row_el = matr[piv_row];
  }
  __syncthreads();

  if (col != piv_col) {
    matr[row] += piv_row_el * cache[row];
  }
  __syncthreads();
  matrix.e[row + col * matrix.m] = matr[row];
}/**/

int gpuDualSimplexSyncDev (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  memInit(matrix);

  std::ifstream input("PivRowCol.txt");

  while (true) {
    flag ++;

    //std::cout << flag << std::endl;
    if (flag % 10000000000 == 0) {
      memFree();
      return 0;
    }

    const int threads = matrix.rows; //(matrix.rows - 1 < SHARED_MEM_SIZE ? matrix.rows - 1 : SHARED_MEM_SIZE);
    pivotRowColumn <<<1, threads>>> (dev_matrix, dev_piv_row, dev_piv_col);
    int pivot_row = 0;
    CHECK_CUDA(cudaMemcpy (&pivot_row, dev_piv_row, sizeof (int) * 1, cudaMemcpyDeviceToHost));
    if (!pivot_row) {
      memFree();
      return flag;
    }

    int pivot_col = 0;
    CHECK_CUDA(cudaMemcpy (&pivot_col, dev_piv_col, sizeof (int) * 1, cudaMemcpyDeviceToHost));
    if (!pivot_col) {
      memFree();
      return 0;
    }

    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    simplexMatrix<<<matrix.cols,matrix.rows>>>(dev_matrix, pivot_row, pivot_col, dev_er);


  }
}
