/**
 * Copyright (c) 2016 ISP RAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
      delete [] result;
      return 0;
    }
  }
  delete [] result;
  return 1;
}
