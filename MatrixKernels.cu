#include "Matrix.h"

__global__
void iden_matr(d_matrix matrix) {
  register int i = threadIdx.x;
  register int j = blockIdx.x;
  register int box = i + j * matrix.m;
  if (i != j) {
    matrix.e[box] = 0.0;
  } else {
    matrix.e[box] = 1.0;
  }
}

__global__
void copyMatrix(d_matrix left, d_matrix right) {
  int l_num = threadIdx.x + blockIdx.x * left.m;
  int r_num = threadIdx.x + blockIdx.x * right.m;
  left.e[l_num] = right.e[r_num];
}

