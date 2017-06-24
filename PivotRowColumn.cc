#include "PivotRowColumn.h"
#include "Epsilon.h"

int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    if (cmp(matrix.e[i + 0 * matrix.m], 0) == -1) {
      return i;
    }
  }
  return 0;
}

int pivotColumn(Matrix const &matrix, const int row) {
  int ret = 0;

  for (int j = 1; j < matrix.cols; j++) {
    if (cmp(matrix.e[row + j * matrix.m], 0) == -1) {
      if (ret == 0) {
        ret = j;
      } else {
        for (int i = 0; i < matrix.rows; i++) {
          double val1 = - matrix.e[i + ret * matrix.m] / matrix.e[row + ret * matrix.m];
          double val2 = - matrix.e[i + j * matrix.m] / matrix.e[row + j * matrix.m];
          int c = cmp(val1,val2);
          if (c == 1) {
            ret = j;
            break;
          }
          if (c == -1) {
            break;
          }
        }
      }
    }
  }
  return ret;
}
