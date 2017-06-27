#include "MatrixTransformation.h"
#include "Epsilon.h"

__global__
void matrixTransform(d_matrix matrix, int piv_row, double *col_e) {
  __shared__ double cache[TRANSFORM_BLOCK_SIZE];
  __shared__ double matr[TRANSFORM_BLOCK_SIZE];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  int size = blockDim.x;

  if (threadIdx.x == 0) {
    piv_row_el = matrix.e[piv_row + col * matrix.m];
  }
  __syncthreads();
  for (int i_row = row; i_row < matrix.rows; i_row += size) {
      cache[row] = col_e[i_row];
      matr[row] = matrix.e[i_row + col * matrix.m];
      matr[row] += piv_row_el * cache[row];
      matrix.e[i_row + col * matrix.m] = matr[row];
  }
  __syncthreads();
}

__global__
void matrixTransformSync(d_matrix matrix, int piv_row, int piv_col, double *col_e) {
  __shared__ double cache[TRANSFORM_BLOCK_SIZE];
  __shared__ double matr[TRANSFORM_BLOCK_SIZE];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x;
  int size = blockDim.x;

  if (threadIdx.x == 0) {
    piv_row_el = matrix.e[piv_row + col * matrix.m];
  }
  __syncthreads();
    for (int i_row = row; i_row < matrix.rows; i_row += size) {
        cache[row] = col_e[i_row];
        if (col != piv_col) {
          matr[row] = matrix.e[i_row + col * matrix.m];
          matr[row] += piv_row_el * cache[row];
        } else {
          matr[row] = cache[row];
        }
        matrix.e[i_row + col * matrix.m] = matr[row];
  }
  __syncthreads();
}

__global__
void matrixTransformDev(d_matrix matrix, int piv_row, int piv_col, double *dev_col,
    int piv_row_next, int *num_cols) {

  __shared__ double cache[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double col_e[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double piv_row_el[BLOCK_SIZE];
  __shared__ double piv_cols[BLOCK_SIZE];
  int row_box = threadIdx.x;
  int col_box = threadIdx.y;
  int el_box = row_box + col_box * BLOCK_SIZE;
  int bound = (matrix.rows - 1) / BLOCK_SIZE + 1;

  int col = 1 + col_box + blockIdx.x * BLOCK_SIZE;
  if (threadIdx.x == 0) {
    piv_row_el[col_box] = matrix.e[piv_row + col * matrix.m];
  }
  __syncthreads();
  //=================transform box which include piv_row_next====================
  int piv_row_box = piv_row_next / BLOCK_SIZE;
  int row = row_box + piv_row_box * BLOCK_SIZE;
  cache[el_box] = matrix.e[row + col * matrix.m];
  col_e[el_box] = dev_col[row]; //????
  if (col != piv_col) {
    cache[el_box] += piv_row_el[col_box] * col_e[el_box];
  } else {
    cache[el_box] = col_e[el_box];
  }
  matrix.e[row + col * matrix.m] = cache[el_box];
  if (row == piv_row_next) {
    piv_cols[col_box] = (cache[el_box] < -EPSILON ? - cache[el_box] : 0.0);
    if (col >= matrix.cols) {
      piv_cols[col_box] = 0.0;
    }
  }
  __syncthreads();

  //================transform whole matrix and choose min col=======================
  row = row_box;
  for (int i = 0; i < bound; i++) {
    if (i == piv_row_box) {
      //=============has been already transformed, choose min col ========================
      cache[el_box] = matrix.e[row + col * matrix.m];
      if (piv_cols[col_box] > EPSILON) {
        cache[el_box] /= piv_cols[col_box];
      }
      __syncthreads();
      if ((piv_cols[col_box] > EPSILON) && (piv_cols[row_box] > EPSILON)) {
        for (int k = 0; k < BLOCK_SIZE; k++) {
          if (cache[k + col_box * BLOCK_SIZE] < cache[k + row_box * BLOCK_SIZE] - EPSILON) {
            piv_cols[row_box] = 0.0;
            break;
          }
          if (cache[k + col_box * BLOCK_SIZE] > cache[k + row_box * BLOCK_SIZE] + EPSILON) {
            piv_cols[col_box] = 0.0;
            break;
          }
        }
      }
    } else {
      //=========================transform===========================================
      cache[el_box] = matrix.e[row + col * matrix.m];
      col_e[el_box] = dev_col[row];
      if (col != piv_col) {
        cache[el_box] += piv_row_el[col_box] * col_e[el_box];
      } else {
        cache[el_box] = col_e[el_box];
      }
      matrix.e[row + col * matrix.m] = cache[el_box];
      if (piv_cols[col_box] > EPSILON) {
        cache[el_box] /= piv_cols[col_box];
      }
      __syncthreads();
      //========================choose min col===========================================
      if ((piv_cols[col_box] > EPSILON) && (piv_cols[row_box] > EPSILON)) {
        for (int k = 0; k < BLOCK_SIZE; k++) {
          if (cache[k + col_box * BLOCK_SIZE] < cache[k + row_box * BLOCK_SIZE] - EPSILON) {
            piv_cols[row_box] = 0.0;
            break;
          }
          if (cache[k + col_box * BLOCK_SIZE] > cache[k + row_box * BLOCK_SIZE] + EPSILON) {
            piv_cols[col_box] = 0.0;
            break;
          }
        }
      }
    }
    __syncthreads();

    row += BLOCK_SIZE;
  }

  if (el_box == 0) {
    for (int j = 0; j < BLOCK_SIZE;j++) {
      if (piv_cols[j] > EPSILON) {
        num_cols[blockIdx.x] = col - col_box + j;
        break;
      }
    }
  }
  __syncthreads();
}
