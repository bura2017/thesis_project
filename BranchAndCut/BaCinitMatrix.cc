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

#include "BranchAndCut.h"
#include "../MatrixMultip/MatrixMultip.h"

#include <iomanip>

void initMatrix(Matrix &matrix, const Matrix &input, taskTree * &task, d_matrix dev_trans) {
  CHECK_NULL(task);
  matrix = input;

  int *branches = new int[matrix.cols];
  for (int i = 0; i < matrix.cols; i++) {
    branches[i] = 0;
  }
  int cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    branches[branch->point] = branch->num * 2 - 1;
    cut_rows++;
    if (branch->cuts != NULL) {
      cut_rows++;
    }
  }
  delete [] branches;

  if (!cut_rows) {
    CHECK_NULL(NULL);
  }
  Matrix cuts (cut_rows, input.cols);
  cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    int point = branch->point;
    double value = branch->value;
    if (branch->num == 0) {
      cuts.e[cut_rows + point * cuts.m] = 1.0;
      cuts.e[cut_rows + 0 * cuts.m] = value;
    } else {
      cuts.e[cut_rows + point * cuts.m] = -1.0;
      cuts.e[cut_rows + 0 * cuts.m] = - value;
    }
    cut_rows++;
    if (branch->cuts != NULL) {
      for (int j = 0; j < matrix.cols; j++) {
        cuts.e[cut_rows + j * cuts.m] = branch->cuts[j];
      }
      cut_rows ++;
    }
  }
  Matrix temp_matrix(cuts.rows, cuts.cols);
  MatMul(cuts, dev_trans, temp_matrix);

  matrix.add_cuts(temp_matrix);
}
