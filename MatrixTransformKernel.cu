#include "MatrixTransformation.h"

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
