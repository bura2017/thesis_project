#include "HandleError.h"
#include "Epsilon.h"

int checkCorrect (Matrix &input, Matrix &output) {
  float *result = new float [output.cols - 1];
  for (int i = 1; i < output.cols; i++) {
    result[i - 1] = output.e[i];
  }
  for (int i = input.cols; i < input.rows; i++) {
    float check = 0.0;
    for (int j = 1; j < input.cols; j++) {
      check += result[j - 1] * input.e[i + j * input.m];
    }
    if (cmp(check, input.e[i + 0 * input.m]) == 1) {
      //std::cout << i << std::endl;
      delete [] result;
      return 0;
    }
  }
  delete [] result;
  return 1;
}
