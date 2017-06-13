#include "MatrixMultip.h"

void multip(Matrix const &left, Matrix const &right, Matrix &answ) {
  CHECK_NULL(left.e);
  CHECK_NULL(right.e);

  if ((&left == &answ) || (&right == &answ)) {
    ERROR ("answer matrix could't be changed");
  }
  if (left.cols != right.rows) {
    ERROR("matrices couldn't be multiplied");
  }
  if ((answ.rows != left.rows) || (answ.cols != right.cols)) {
    ERROR("reinit");
  }
  for (int j = 0; j < answ.cols; j++) {
    for (int i = 0; i < answ.rows; i++) {
      answ.e[i + j * answ.m] = 0.0;
      for (int k = 0; k < left.cols; k ++){
        answ.e[i + j * answ.m] += left.e[i + k * left.m] * right.e[k + j * right.m];

      }
    }
  }
}
