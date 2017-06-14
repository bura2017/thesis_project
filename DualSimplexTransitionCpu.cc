#include "DualSimplex.h"
#include <iostream>

int cpuDualSimplex (Matrix &matrix, Matrix &transition) {
  CHECK_NULL(matrix.e);
  CHECK_NULL(transition.e);

  int flag = 0;
  Matrix input(matrix);

  if ((transition.rows != transition.cols) || (transition.rows != matrix.cols)) {
    //std::cout << transition.rows << ' ' << transition.cols << ' ' << matrix.cols << std::endl;
    ERROR ("reinit");
  }
  Matrix temp_trans_1(transition.rows, transition.cols);
  Matrix temp_trans_2(transition.rows, transition.cols);
  for (int i = 0; i < transition.rows; i++) {
    temp_trans_1.e[i + i * temp_trans_1.m] = 1.0;
    temp_trans_2.e[i + i * temp_trans_2.m] = 1.0;
  } //identity matrix

  while (1) {
    flag ++;
    if (flag % 1000000 == 0) {
      //return false;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      transition = ((flag % 2) ? temp_trans_1 : temp_trans_2);
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      return 0;
    }
    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    Matrix right_temp(transition.rows, transition.cols);
    for (int i = 0; i < right_temp.rows; i++) {
      right_temp.e[i + i * right_temp.m] = 1.0;
    } //identity matrix
    right_temp.e[pivot_col + pivot_col * right_temp.m] /= - matrix.e[pivot_row + pivot_col * matrix.m];

    for (int j = 0; j < right_temp.cols; j++) {
      if (j != pivot_col) {
        right_temp.e[pivot_col + j * right_temp.m] = matrix.e[pivot_row + j * matrix.m] * right_temp.e[pivot_col + pivot_col * right_temp.m];

      }
    }
    if (flag % 2) {
      multip (temp_trans_1, right_temp, temp_trans_2);
    } else {
      multip (temp_trans_2, right_temp, temp_trans_1);
    }

    int err = matrixTransformCpu (matrix, pivot_row, pivot_col);
    if (err) {
      std::cout << "flag = " << flag << ", col =  " << err << std::endl;
      ERROR ("negative function");
    }

  }

  return 0;
}

