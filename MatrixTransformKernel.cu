#include "MatrixTransformation.h"

__global__
void matrixTransform(d_matrix matrix, int piv_row, double *col_e) {
  __shared__ double cache[1025];
  __shared__ double matr[1025];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  //one column for one block

  cache[row] = col_e[row];
  matr[row] = matrix.e[row + col * matrix.m];
  if (threadIdx.x == piv_row) {
    piv_row_el = matr[piv_row];
  }
  __syncthreads();
  matr[row] += piv_row_el * cache[row];
  __syncthreads();
  matrix.e[row + col * matrix.m] = matr[row];
}
