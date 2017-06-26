#include "MatrixTransformation.h"


int matrixTransformSync(Matrix &matrix, const int row, const int col, d_matrix &dev_matrix, double *dev_col) {
  double div = - matrix.e[row + col * matrix.m];
  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }
  CHECK_CUDA (cudaMemcpy (dev_col, &matrix.e[0 + col * matrix.m], sizeof (double) * matrix.rows,
      cudaMemcpyHostToDevice));

  matrixTransformSync<<<matrix.cols, TRANSFORM_BLOCK_SIZE>>>(dev_matrix, row, col, dev_col);

  CHECK_CUDA (cudaMemcpy (matrix.e, dev_matrix.e, sizeof (double) * matrix.m * matrix.cols,
      cudaMemcpyDeviceToHost));

  return 0;
}
