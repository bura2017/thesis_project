#include "MatrixTransformation.h"


int matrixTransformSync(Matrix &matrix, const int row, const int col, d_matrix &dev_matrix, float *dev_col) {
  float div = - matrix.e[row + col * matrix.m];
  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }
  CHECK_CUDA (cudaMemcpy (dev_matrix.e, matrix.e, sizeof (float) * matrix.m * matrix.cols,
      cudaMemcpyHostToDevice));
  CHECK_CUDA (cudaMemcpy (dev_col, &matrix.e[0 + col * matrix.m], sizeof (float) * matrix.rows,
      cudaMemcpyHostToDevice));

  matrixTransform<<<matrix.cols, matrix.rows>>>(dev_matrix, row, dev_col);

  CHECK_CUDA (cudaMemcpy (matrix.e, dev_matrix.e, sizeof (float) * matrix.m * matrix.cols,
      cudaMemcpyDeviceToHost));

  for (int j = 0; j < matrix.cols; j++) {
    matrix.e[row + j * matrix.m] = (j == col ? -1.0 : 0.0);
  }
  return 0;
}
