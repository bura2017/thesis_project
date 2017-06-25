#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include "HandleError.h"
#include "Epsilon.h"

Matrix::Matrix (int rows, int cols, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply), e(new float[m * cols]) {
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
  //std::cout << cols << ' ' << rows << std::endl;
  e = new float[m * cols];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float num;
      float den = 1.0;
      input >> num >> den;
      e[i + j * m] = (float) num / den;
    }
  }
}
Matrix::Matrix(Matrix const &input, int supply) :
    rows(input.rows), cols(input.cols), supply(input.supply + supply), m(input.m + supply), e(new float[m * cols]) {
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
    rows(input.rows), cols(input.cols), supply(input.m - rows), m(input.m), e(new float[m * cols]) {
  CHECK_CUDA(cudaMemcpy (e, input.e, sizeof(float) * m * cols, cudaMemcpyDeviceToHost));
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
