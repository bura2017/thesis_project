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

#include "../HandleError.h"
#include "MatrixMultip.h"

#include <curand.h>
#include <cublas_v2.h>

void cublas_multip (Matrix const &left, Matrix const &right, Matrix &answ) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);//default
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);// not allowed id default

  d_matrix dev_left;
  dev_left.rows = left.rows;
  dev_left.cols = left.cols;
  dev_left.m = left.m;
  CHECK_CUDA(cudaMalloc(&dev_left.e, sizeof (double) * dev_left.m * dev_left.cols));
  CHECK_CUBLAS(cublasSetMatrix(left.rows, left.cols, sizeof(double), left.e, left.m,
      dev_left.e, dev_left.m));

  d_matrix dev_right;
  dev_right.rows = right.rows;
  dev_right.cols = right.cols;
  CHECK_CUDA(cudaMalloc(&dev_right.e, sizeof (double) * dev_right.rows * dev_right.cols));
  CHECK_CUBLAS(cublasSetMatrix(right.rows, right.cols, sizeof(double), right.e, right.rows,
      dev_right.e, dev_right.rows));

  d_matrix dev_answ;
  dev_answ.rows = answ.rows;
  dev_answ.cols = answ.cols;
  dev_answ.m = answ.m;
  CHECK_CUDA(cudaMalloc(&dev_answ.e, sizeof (double) * dev_answ.m * dev_answ.cols));
  double a = 1.0, b = 0.0;

  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dev_left.rows, dev_right.cols, dev_left.cols,
      &a, dev_left.e, dev_left.m, dev_right.e, dev_right.m, &b, dev_answ.e, dev_answ.m));

  CHECK_CUBLAS(cublasGetMatrix(answ.rows, answ.cols, sizeof(double), dev_answ.e, dev_answ.rows, answ.e, answ.rows));

  cublasDestroy(handle);
  cudaFree(dev_left.e);
  cudaFree(dev_right.e);
  cudaFree(dev_answ.e);

}
void cublas_multip (Matrix const &left, d_matrix const &dev_right, Matrix &answ) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);//default
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);// not allowed id default

  d_matrix dev_left;
  dev_left.rows = left.rows;
  dev_left.cols = left.cols;
  dev_left.m = left.m;
  CHECK_CUDA(cudaMalloc(&dev_left.e, sizeof (double) * dev_left.m * dev_left.cols));
  CHECK_CUBLAS(cublasSetMatrix(left.rows, left.cols, sizeof(double), left.e, left.m,
      dev_left.e, dev_left.m));

  d_matrix dev_answ;
  dev_answ.rows = answ.rows;
  dev_answ.cols = answ.cols;
  dev_answ.m = answ.m;
  CHECK_CUDA(cudaMalloc(&dev_answ.e, sizeof (double) * dev_answ.m * dev_answ.cols));
  double a = 1.0, b = 0.0;

  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dev_left.rows, dev_right.cols, dev_left.cols,
      &a, dev_left.e, dev_left.m, dev_right.e, dev_right.m, &b, dev_answ.e, dev_answ.m));

  CHECK_CUBLAS(cublasGetMatrix(answ.rows, answ.cols, sizeof(double), dev_answ.e, dev_answ.rows, answ.e, answ.rows));

  cublasDestroy(handle);

  cudaFree(dev_left.e);
  cudaFree(dev_answ.e);
}
void cublas_multip (d_matrix &dev_left, d_matrix &dev_right, d_matrix &dev_answ, cudaStream_t stream) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);//host is default
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);// not allowed is default

  double a = 1.0, b = 0.0;

  cublasSetStream(handle, stream);
  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dev_left.rows, dev_right.cols, dev_left.cols,
      &a, dev_left.e, dev_left.m, dev_right.e, dev_right.m, &b, dev_answ.e, dev_answ.m));

  cublasDestroy(handle);

}
void cublas_multip (d_matrix &dev_left, d_matrix &dev_right, d_matrix &dev_answ) {
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);//host is default
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);// not allowed is default

  double a = 1.0, b = 0.0;

  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dev_left.rows, dev_right.cols, dev_left.cols,
      &a, dev_left.e, dev_left.m, dev_right.e, dev_right.m, &b, dev_answ.e, dev_answ.m));

  cublasDestroy(handle);

}
