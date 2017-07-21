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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef MATRIX_H_
#define MATRIX_H_

#define MAX_BLOCKS 128
#define MAX_LENG 1024
#define BLOCK_SIZE 16
#define TRANSFORM_BLOCK_SIZE 128
#define CPU_COLS 1

#define GCD(x,y) {double a = x;\
  double b = y;                \
  if (a < 0) {                 \
    a = -a;                    \
  }                            \
  if (a < epsilon) {           \
    a = b;                     \
  } else {                     \
  if (b < 0) {                 \
    b = -b;                    \
  }                            \
  while (b > epsilon) {        \
    double temp = b;           \
    b = a - b * floor(a / b);  \
    a = temp;                  \
  }                            \
  }                            \
  x /= a;                      \
  y /= a;}

typedef struct {
    int rows;
    int cols;
    int m;
    double *e;
} d_matrix;


struct Matrix {
  int rows;
  int cols;
  int supply;
  int m;
  double *e;

  Matrix(int rows, int cols, int supply = 0);
  Matrix (int rows, int cols, unsigned int flag, int supply = 0);
  Matrix(char const *file_name, int supply = 0);
  Matrix(Matrix const &input, int supply = 0);
  Matrix(Matrix const &input, unsigned int flag, int supply = 0);
  Matrix(d_matrix const &input);
  ~Matrix();
  void freeHost();
  int print(char const *filename) const;
  int print_task(char const *filename) const;
  int add_cuts(Matrix const &cuts);
  Matrix &operator=(Matrix const &matrix);
};

__global__ void iden_matr(d_matrix matrix);
__global__ void copyMatrix(d_matrix left, d_matrix right);


#endif /* MATRIX_H_ */
