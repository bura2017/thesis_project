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

#include <iostream>

void printError(const char *error, const char *file, int line) {
  std::cout <<"ERROR " << error << " in " << file << " at line " << line << std::endl;
  exit( EXIT_FAILURE );
}
void HandleCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printError (cudaGetErrorString(err), file, line);
  }
}
void HandleNullError(const void *var, const char *file, int line) {
  if (var == NULL) {
    printError ("NULL", file, line);
  }
}
const char* cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}
void HandleCublasError(cublasStatus_t err, const char *file, int line){
  if (err != CUBLAS_STATUS_SUCCESS) {
    printError (cublasGetErrorString(err), file, line);
  }
}
