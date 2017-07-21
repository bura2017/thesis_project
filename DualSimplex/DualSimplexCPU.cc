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

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      return -flag;
    }

    int err = matrixTransformCpu (matrix, pivot_row, pivot_col);
    if (err) {
      return 0;
    }
  }

  return false;
}
