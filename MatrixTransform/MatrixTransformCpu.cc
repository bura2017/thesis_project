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

#include "MatrixTransformation.h"
#include "../Epsilon.h"

#include <iostream>

int matrixTransformCpu(Matrix &matrix, const int row, const int col) {
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
          return j;
        }
      }
    }
  }
  for (int j = 0; j < matrix.cols; j++) {
    matrix.e[row + j * matrix.m] = (j == col ? -1.0 : 0.0);
  }
  return 0;
}
