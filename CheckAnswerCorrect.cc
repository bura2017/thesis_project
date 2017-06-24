#include "HandleError.h"
#include "Epsilon.h"

int checkCorrect (Matrix &input, Matrix &output) {
  double epsilon = 0.1;
  double *result = new double [output.cols - 1];
  for (int i = 1; i < output.cols; i++) {
    result[i - 1] = output.e[i];
  }
  for (int i = input.cols; i < input.rows; i++) {
    double check = 0.0;
    for (int j = 1; j < input.cols; j++) {
      check += result[j - 1] * input.e[i + j * input.m];
    }
    if (check > input.e[i + 0 * input.m] + epsilon) {
      //std::cout << i << std::endl;
      delete [] result;
      return 0;
    }
  }
  delete [] result;
  return 1;
}
