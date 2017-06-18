#include "MatrixTransformation.h"
#include "Epsilon.h"
#include <iostream>

int matrixTransformCpu(Matrix &matrix, const int row, const int col) {
  double div = - matrix.e[row + col * matrix.m];

  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }

  for(int j = 0; j < matrix.cols; j++) {
    for(int i = 0; i < matrix.rows; i++) {
      if ((j != col) && (i != row)) {
        int box = i + j * matrix.m;
        matrix.e[box] += matrix.e[i + col * matrix.m] * matrix.e[row + j * matrix.m];
        if ((i == 0) && (j) && (cmp(matrix.e[box], 0) == -1)) {
          return j;
        }
      }
    }
  }
  for (int j = 0; j < matrix.cols; j++) {
    matrix.e[row + j * matrix.m] = (j == col ? -1.0 : 0.0);
  }
  return 0;
}
