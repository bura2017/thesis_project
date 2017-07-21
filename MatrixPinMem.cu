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

#include "Matrix.h"

Matrix::Matrix (int rows, int cols, unsigned int flag, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply) {
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  CHECK_CUDA(cudaHostAlloc((void**)&e, m * cols * sizeof(double), flag));
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = 0.0;
    }
  }
}

Matrix::Matrix(Matrix const &input, unsigned int flag, int supply) :
    rows(input.rows), cols(input.cols), supply(supply), m(input.m + supply) {
  const uint full = m * cols;
  CHECK_CUDA(cudaHostAlloc((void**)&e, full * sizeof(double), flag));

  for (int j = 0; j < cols; j ++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = input.e[i + j * input.m];
    }
  }
}

void Matrix::freeHost() {
  CHECK_NULL(e);

  cudaFreeHost(e);
  e = NULL;
}
