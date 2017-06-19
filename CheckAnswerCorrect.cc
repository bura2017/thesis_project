#include "HandleError.h"
#include "Epsilon.h"

int checkCorrect (Matrix &input, Matrix &output) {
  double *result = new double [output.cols];
  for (int i = 1; i < output.cols; i++) {
    result[i] = output.e[i];
  }
  for (int i = input.cols; i < input.rows; i++) {
    double check = 0.0;
    for (int j = 1; j < input.cols; j++) {
      check += result[j] * input.e[i + j * input.m];
    }
    if (cmp(check, input.e[i + 0 * input.m]) == 1) {
      delete [] result;
      return 0;
    }
  }
  delete [] result;
  return 1;
}
