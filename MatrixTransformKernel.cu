#include "MatrixTransformation.h"

__global__
void matrixTransform(d_matrix matrix, int piv_row, float *col_e) {
  __shared__ float cache[1025];
  __shared__ float matr[1025];
  __shared__ float piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  int size = blockDim.x;
  int bound = (matrix.rows - 1) / size + 1;

  if (threadIdx.x == 0) {
    piv_row_el = matrix.e[piv_row + col * matrix.m];
  }
  __syncthreads();
  for (int i = 0; i < bound; i++) {
    int i_row = row + i * size;
    if (i_row < matrix.rows) {
      cache[row] = col_e[i_row];
      matr[row] = matrix.e[i_row + col * matrix.m];
      matr[row] += piv_row_el * cache[row];
      matrix.e[i_row + col * matrix.m] = matr[row];
    }
    __syncthreads();
  }
}
