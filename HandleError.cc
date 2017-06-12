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
