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

int matrixTransformSync(Matrix &matrix, const int row, const int col, d_matrix &dev_matrix, double *dev_col) {
  double div = - matrix.e[row + col * matrix.m];
  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }
  CHECK_CUDA (cudaMemcpy (dev_col, &matrix.e[0 + col * matrix.m], sizeof (double) * matrix.rows,
      cudaMemcpyHostToDevice));

  matrixTransformSync<<<matrix.cols, TRANSFORM_BLOCK_SIZE>>>(dev_matrix, row, col, dev_col);

  CHECK_CUDA (cudaMemcpy (matrix.e, dev_matrix.e, sizeof (double) * matrix.m * matrix.cols,
      cudaMemcpyDeviceToHost));

  return 0;
}
