#include "DualSimplex.h"
#include <fstream>
#include <iostream>
#include <iomanip>

static int flag;
static int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    if (cmp(matrix.e[i + 0 * matrix.m], 0) == -1) {
      return i;
    }
  }
  return 0;
}
static int pivotColumn(Matrix const &matrix, const int row) {
  int ret = 0;

  for (int j = 1; j < matrix.cols; j++) {
    if (cmp(matrix.e[row + j * matrix.m], 0) == -1) {
      if (ret == 0) {
        ret = j;
      } else {
        for (int i = 0; i < matrix.rows; i++) {
          double val1 = matrix.e[i + ret * matrix.m] * matrix.e[row + j * matrix.m];
          double val2 = matrix.e[i + j * matrix.m] * matrix.e[row + ret * matrix.m];
          int c = cmp(val1,val2);
          if (c == -1) {
            ret = j;
            break;
          }
          if (c == 1) {
            break;
          }
        }
      }
    }
  }
  return ret;
}
static void dualSimplex(Matrix &matrix, const int row, const int col) {
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
          std::cout << "flag = " << flag << ", col =  " << j << std::endl;
          matrix.print("Check.txt");
          ERROR ("negative function");
        }
      }
    }
  }
  for (int j = 0; j < matrix.cols; j++) {
    matrix.e[row + j * matrix.m] = (j == col ? -1.0 : 0.0);
  }
}
int cpuDualSimplex (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  flag = 0;
  //std::ofstream output("PivRowCol.txt");

  while (1) {
    flag ++;
    if (flag % 1000000 == 0) {
      return 0;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      return 0;
    }

    //std::cout << "flag = " << flag << ", row = " << pivot_row << ", col = " << pivot_col << std::endl;

    dualSimplex (matrix, pivot_row, pivot_col);

  }

  return false;
}

//========================================================================================================================================================================

int cpuDualSimplex (Matrix &matrix, Matrix &transition) {
  CHECK_NULL(matrix.e);
  CHECK_NULL(transition.e);

  flag = 0;
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

    dualSimplex (matrix, pivot_row, pivot_col);
  }

  return 0;
}
