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

#include "TransitionMatrix.h"
#include "../Epsilon.h"

int dev_trans_init(d_matrix &dev_trans, Matrix &input) {
  const int side = input.cols;
  dev_trans.rows = side;
  dev_trans.cols = side;
  dev_trans.m = side + BLOCK_SIZE - 1;
  CHECK_CUDA(cudaMalloc(&dev_trans.e, sizeof(double) * dev_trans.m * dev_trans.m));

  Matrix trans(dev_trans.m, dev_trans.m);
  for (int i = 0; i < dev_trans.m; i++) {
    trans.e[i + i * trans.m] = 1.0;
  }//identity mantrix
  for (int j = 0; j < side; j++) {
    for (int i = 1; i < side; i++) {
      trans.e[i + j * trans.m] = - input.e[i + j * input.m];
      if (cmp(input.e[i + j * input.m], 0.0) == 0) {
        trans.e[i + j * trans.m] = 0.0;
      }
    }
  }
  CHECK_CUDA(cudaMemcpy (dev_trans.e, trans.e, sizeof(double) * dev_trans.m * dev_trans.m, cudaMemcpyHostToDevice));

  return 0;
}
int dev_trans_free(d_matrix &dev_trans) {
  cudaFree(dev_trans.e);
  return 0;
}
