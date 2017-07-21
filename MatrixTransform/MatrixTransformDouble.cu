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

static int getSizes (int prev, int col, const int max_col, int &cur, int &blocks) {
  if (prev == max_col) {
    cur = 0;
    blocks = 0;
    return 1;
  }
  cur = prev;
  blocks = MAX_BLOCKS;
  if (cur < col) {
    if (cur + blocks > col) {
      blocks = col - cur;
    }
    return 0;
  }
  if (cur == col) {
    cur++;
  }
  if (cur > col) {
    if (cur + blocks > max_col) {
      blocks = max_col - cur;
    }
    return 0;
  }
  return 0;
}

int matrixTransformDouble (data_full_task data0, data_full_task data1) {
  double div_0 = - data0.matrix->e[data0.piv_row + data0.piv_col * data0.matrix->m];
  double div_1 = - data1.matrix->e[data1.piv_row + data1.piv_col * data1.matrix->m];

  for (int i = 0; i < data0.matrix->rows; i++) {
    data0.matrix->e[i + data0.piv_col * data0.matrix->m] /= div_0;
  }

  for (int i = 0; i < data1.matrix->rows; i++) {
    data1.matrix->e[i + data1.piv_col * data1.matrix->m] /= div_1;
  }

  CHECK_CUDA (cudaMemcpyAsync (data0.dev_col, &(data0.matrix->e[0 + data0.piv_col * data0.matrix->m]), sizeof(double) * data0.matrix->rows,
      cudaMemcpyHostToDevice, data0.stream));
  CHECK_CUDA (cudaMemcpyAsync (data1.dev_col, &(data1.matrix->e[0 + data1.piv_col * data1.matrix->m]), sizeof(double) * data1.matrix->rows,
      cudaMemcpyHostToDevice, data1.stream));

  int innokentiy[] = {0, 0};
  int blocks[] = {0, 0};

  while (true) {
    int c = getSizes (innokentiy[0] + blocks[0], data0.piv_col, data0.matrix->cols, innokentiy[0], blocks[0]);
    c += getSizes (innokentiy[1] + blocks[1], data1.piv_col, data1.matrix->cols, innokentiy[1], blocks[1]);
    //std::cout << "0: " << innokentiy[0] << ' ' << blocks[0] << std::endl;
    //std::cout << "1: " << innokentiy [1] << ' ' << blocks[1] << std::endl;

    if (c) {
      break;
    }
    CHECK_NULL(memcpy (data0.pin_matrix->e, &(data0.matrix->e[0 + innokentiy[0] * data0.matrix->m]),
        sizeof(double) * blocks[0] * data0.matrix->m));
    CHECK_NULL(memcpy (data1.pin_matrix->e, &(data1.matrix->e[0 + innokentiy[1] * data1.matrix->m]),
        sizeof(double) * blocks[1] * data1.matrix->m));

    CHECK_CUDA(cudaMemcpyAsync(data0.dev_matrix.e, data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyHostToDevice, data0.stream));
    CHECK_CUDA(cudaMemcpyAsync(data1.dev_matrix.e, data1.pin_matrix->e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyHostToDevice, data1.stream));

    matrixTransform<<<blocks[0],TRANSFORM_BLOCK_SIZE,0,data0.stream>>>(data0.dev_matrix, data0.piv_row, data0.dev_col);
    matrixTransform<<<blocks[1],TRANSFORM_BLOCK_SIZE,0,data1.stream>>>(data1.dev_matrix, data1.piv_row, data1.dev_col);

    CHECK_CUDA(cudaMemcpyAsync(data0.pin_matrix->e, data0.dev_matrix.e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyDeviceToHost, data0.stream));
    CHECK_CUDA(cudaMemcpyAsync(data1.pin_matrix->e, data1.dev_matrix.e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyDeviceToHost, data1.stream));


    CHECK_CUDA(cudaStreamSynchronize(data0.stream));
    CHECK_NULL(memcpy (&(data0.matrix->e[0 + innokentiy[0] * data0.matrix->m]), data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.matrix->m));

    CHECK_CUDA(cudaStreamSynchronize(data1.stream));
    CHECK_NULL(memcpy (&(data1.matrix->e[0 + innokentiy[1] * data1.matrix->m]), data1.pin_matrix->e,
        sizeof(double) * blocks[1] * data1.matrix->m));

  }
  if (innokentiy[0]) {
    CHECK_NULL(memcpy (data0.pin_matrix->e, &(data0.matrix->e[0 + innokentiy[0] * data0.matrix->m]),
        sizeof(double) * blocks[0] * data0.matrix->m));
    CHECK_CUDA(cudaMemcpyAsync(data0.dev_matrix.e, data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyHostToDevice, data0.stream));
    matrixTransform<<<blocks[0],TRANSFORM_BLOCK_SIZE,0,data0.stream>>>(data0.dev_matrix, data0.piv_row, data0.dev_col);
    CHECK_CUDA(cudaMemcpyAsync(data0.pin_matrix->e, data0.dev_matrix.e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyDeviceToHost, data0.stream));
    CHECK_CUDA(cudaStreamSynchronize(data0.stream));
    CHECK_NULL(memcpy (&(data0.matrix->e[0 + innokentiy[0] * data0.matrix->m]), data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.matrix->m));
  }
  if (innokentiy[1]) {
    CHECK_NULL(memcpy (data1.pin_matrix->e, &(data1.matrix->e[0 + innokentiy[1] * data1.matrix->m]),
        sizeof(double) * blocks[1] * data1.matrix->m));
    CHECK_CUDA(cudaMemcpyAsync(data1.dev_matrix.e, data1.pin_matrix->e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyHostToDevice, data1.stream));
    matrixTransform<<<blocks[1],TRANSFORM_BLOCK_SIZE,0,data1.stream>>>(data1.dev_matrix, data1.piv_row, data1.dev_col);
    CHECK_CUDA(cudaMemcpyAsync(data1.pin_matrix->e, data1.dev_matrix.e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyDeviceToHost, data1.stream));
    CHECK_CUDA(cudaStreamSynchronize(data1.stream));
    CHECK_NULL(memcpy (&(data1.matrix->e[0 + innokentiy[1] * data1.matrix->m]), data1.pin_matrix->e,
        sizeof(double) * blocks[1] * data1.matrix->m));
  }

  return 0;
}
