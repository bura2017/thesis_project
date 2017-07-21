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

#include "Matrix.h"
#include "HandleError.h"
#include "Epsilon.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <cmath>

Matrix::Matrix (int rows, int cols, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply), e(new double[m * cols]) {
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  for (int j = 0; j < this->cols; j++) {
    for (int i = 0; i < m; i++) {
      e[i + j * m] = 0.0;
    }
  }
}
Matrix::Matrix(char const *filename, int supply) : rows(0), cols(0), supply(supply) {
  CHECK_NULL(filename);

  std::ifstream input(filename);
  if (!input.is_open()) {
    ERROR("file isn't open");
  }
  input >> cols >> rows;
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  m = rows + supply;
  e = new double[m * cols];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double num;
      double den = 1.0;
      input >> num >> den;
      e[i + j * m] = (double) num / den;
    }
  }
}
Matrix::Matrix(Matrix const &input, int supply) :
    rows(input.rows), cols(input.cols), supply(input.supply + supply), m(input.m + supply), e(new double[m * cols]) {
  for (int j = 0; j < cols; j ++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = input.e[i + j * input.m];
    }
    for (int i = rows; i < m; i++) {
      e[i + j * m] = 0.0;
    }
  }
}
Matrix::Matrix(d_matrix const &input) :
    rows(input.rows), cols(input.cols), supply(input.m - rows), m(input.m), e(new double[m * cols]) {
  CHECK_CUDA(cudaMemcpy (e, input.e, sizeof(double) * m * cols, cudaMemcpyDeviceToHost));
}
Matrix::~Matrix() {
  if (e != NULL) {
    delete [] e;
    e = NULL;
  }
}
int Matrix::print(char const *filename) const {
  CHECK_NULL(e);

  char fullname[MAX_LENG];
  strcpy (fullname, "Out/");
  strcat (fullname, filename);

  std::ofstream output(fullname);
  if (!output.is_open()) {
    ERROR("file isn't open");
  }
  output << rows << ' ' << cols << std::endl;
  for (int i = 0 ; i < rows; i ++) {
    for (int j = 0; j < cols; j ++) {
      output << std::setw(7) << e[i + j * m] << ' ';
    }
    output << std::endl;
  }
  return 0;
}
int Matrix::print_task(char const *filename) const {
  std::ofstream output(filename);
  output << cols << ' ' << rows << std::endl;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output << e[i + j * m] << std::endl;
    }
  }
  return 0;
}
int Matrix::add_cuts(Matrix const &cuts) {
  if (cuts.cols != cols) {
    ERROR("Wrong input data");
  }
  if(supply < cuts.rows) {
    std::cout << supply << ' ' << cuts.rows << std::endl;
    ERROR("reinit");
  }
  supply -= cuts.rows;
  for (int j = 0; j < cuts.cols; j ++ ) {
    for (int i = 0; i < cuts.rows; i++) {
      e[(rows + i) + j * m] = cuts.e[i + j * cuts.m];
    }
  }
  rows += cuts.rows;
  return 0;
}
Matrix &Matrix::operator=(Matrix const &matrix) {
  if (this != &matrix) {
    if (m * cols < matrix.rows * matrix.cols) {
      ERROR("reinit");
    }
    supply = m - matrix.rows;
    rows = matrix.rows;
    cols = matrix.cols;

    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        e[i + j * m] = matrix.e[i + j * matrix.m];
      }
    }
  }
  return *this;
}
