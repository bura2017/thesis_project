#include "DualSimplex.h"
#include <fstream>
#include <iostream>
#include <iomanip>

static int flag;

int cpuDualSimplex (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  flag = 0;
  //std::ofstream output("PivRowCol.txt");

  while (1) {
    flag ++;
    if (flag % 1000000 == 0) {
      std::cout << "ups" << std::endl;
      return 0;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      return -flag;
    }

    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    int err = matrixTransformCpu (matrix, pivot_row, pivot_col);
    if (err) {
      return 0;
    }
  }

  return false;
}
