#include "Matrix.h"

Matrix::Matrix (int rows, int cols, unsigned int flag, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply) {
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  CHECK_CUDA(cudaHostAlloc((void**)&e, m * cols * sizeof(double), flag));
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = 0.0;
    }
  }
}

Matrix::Matrix(Matrix const &input, unsigned int flag, int supply) :
    rows(input.rows), cols(input.cols), supply(supply), m(input.m + supply) {
  const uint full = m * cols;
  CHECK_CUDA(cudaHostAlloc((void**)&e, full * sizeof(double), flag));

  for (int j = 0; j < cols; j ++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = input.e[i + j * input.m];
    }
  }
}

void Matrix::freeHost() {
  CHECK_NULL(e);

  cudaFreeHost(e);
  e = NULL;
}
