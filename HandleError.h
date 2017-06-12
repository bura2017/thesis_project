/*
 * HandleError.h
 *
 *  Created on: 14 февр. 2017 г.
 *      Author: valerius
 */

#ifndef HANDLEERROR_H_
#define HANDLEERROR_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(X) HandleCudaError (X, __FILE__, __LINE__)
#define CHECK_NULL(X) HandleNullError (X, __FILE__, __LINE__)
#define CHECK_CUBLAS(X) HandleCublasError (X, __FILE__, __LINE__)
#define ERROR(X) printError (X, __FILE__, __LINE__)

void printError(const char *error, const char *file, int line);
void HandleCudaError(cudaError_t err, const char *file, int line);
void HandleNullError(const void *var, const char *file, int line);
void HandleCublasError(cublasStatus_t err, const char *file, int line);

#endif /* HANDLEERROR_H_ */
