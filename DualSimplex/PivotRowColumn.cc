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

#include "PivotRowColumn.h"
#include "../Epsilon.h"

int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    if (cmp(matrix.e[i + 0 * matrix.m], 0) == -1) {
      return i;
    }
  }
  return 0;
}

int pivotColumn(Matrix const &matrix, const int row) {
  int ret = 0;

  for (int j = 1; j < matrix.cols; j++) {
    if (cmp(matrix.e[row + j * matrix.m], 0) == -1) {
      if (ret == 0) {
        ret = j;
      } else {
        for (int i = 0; i < matrix.rows; i++) {
          double val1 = - matrix.e[i + ret * matrix.m] / matrix.e[row + ret * matrix.m];
          double val2 = - matrix.e[i + j * matrix.m] / matrix.e[row + j * matrix.m];
          int c = cmp(val1,val2);
          if (c == 1) {
            ret = j;
            break;
          }
          if (c == -1) {
            break;
          }
        }
      }
    }
  }
  return ret;
}
