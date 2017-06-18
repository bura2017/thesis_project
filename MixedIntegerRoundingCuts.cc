#include "BranchAndCut.h"
#include "Epsilon.h"

#define MAX(x,y) (((x) > (y)) ? (x) : (y))

int mirCuts (Matrix &matrix, double *cuts) {
  if (cuts != NULL) {
    std::cout << "ERROR cut already exists" << std::cout;
    return 0;
  }

  double viol_best = 0.0;
  int row_best = 0;

  double *ineq = new double [matrix.cols];
  for (int i = 1; i < matrix.cols; i++) {
    ineq[0] = matrix.e[i + 0 * matrix.m];
    double f_0 = ineq[0] - floor(ineq[0]);
    ineq[0] = floor(ineq[0]);
    for (int j = 1; j < matrix.cols; j++) {
      double a_j = matrix.e[i + j * matrix.m];
      ineq[j] = floor(a_j) + MAX(a_j - floor(a_j) - f_0, 0) / (1 - f_0);
    }

    double viol = 0.0;
    for (int j = 1; j < matrix.cols; j++) {
      viol += matrix.e[j] * ineq[j];
    }
    if (cmp(viol, ineq[0]) == 1) {
      viol = viol - ineq[0];
      if (viol_best < viol) {
        viol_best = viol;
        row_best = i;
      }
    }
  }

  if (row_best > 0) {
    ineq[0] = matrix.e[row_best + 0 * matrix.m];
    double f_0 = ineq[0] - floor(ineq[0]);
    ineq[0] = floor(ineq[0]);
    for (int j = 1; j < matrix.cols; j++) {
      double a_j = matrix.e[row_best + j * matrix.m];
      ineq[j] = floor(a_j) + MAX(a_j - floor(a_j) - f_0, 0) / (1 - f_0);
    }
  }
  if (row_best == 0) {
    delete [] ineq;
  } else {
    cuts = ineq;
  }

  return row_best;
}
