/*
 * Matrix.h
 *
 *  Created on: 14 февр. 2017 г.
 *      Author: valerius
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "HandleError.h"

#ifndef MATRIX_H_
#define MATRIX_H_

#define MAX_BLOCKS 128
#define MAX_LENG 1024
#define BLOCK_SIZE 16
#define TRANSFORM_BLOCK_SIZE 128

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
