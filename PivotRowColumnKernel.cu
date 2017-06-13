#include "PivotRowColumn.h"

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
