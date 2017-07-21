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

#ifndef HANDLEERROR_H_
#define HANDLEERROR_H_

#include "Matrix.h"

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

int checkCorrect (Matrix &input, Matrix &output);

#endif /* HANDLEERROR_H_ */
