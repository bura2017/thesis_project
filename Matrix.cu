#include "Matrix.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include "HandleError.h"

Matrix::Matrix (int rows, int cols, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply), e(new double[m * cols]) {
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = 0.0;
    }
  }
}
Matrix::Matrix (int rows, int cols, unsigned int flag, int supply) :
    rows(rows), cols(cols), supply(supply), m(rows + supply) {
  if (rows == 0 || cols == 0) {
    CHECK_NULL(NULL);
  }
  CHECK_CUDA(cudaHostAlloc((void**)&e, m * cols * sizeof(double), flag));
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
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
  e = new double[m * cols];

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int num;
      int den;
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
Matrix::Matrix(Matrix const &input, unsigned int flag, int supply) :
    rows(input.rows), cols(input.cols), supply(supply), m(input.m + supply) {
  const uint full = m * cols;
  CHECK_CUDA(cudaHostAlloc((void**)&e, full * sizeof(double), flag));

  for (int j = 0; j < cols; j ++) {
    for (int i = 0; i < rows; i++) {
      e[i + j * m] = input.e[i + j * input.m];
    }
  }
}
Matrix::~Matrix() {
  if (e != NULL) {
    delete [] e;
    e = NULL;
  }
}
void Matrix::freeHost() {
  CHECK_NULL(e);

  cudaFreeHost(e);
  e = NULL;
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

void multip(Matrix const &left, Matrix const &right, Matrix &answ) {
  CHECK_NULL(left.e);
  CHECK_NULL(right.e);

  if ((&left == &answ) || (&right == &answ)) {
    ERROR ("answer matrix could't be changed");
  }
  if (left.cols != right.rows) {
    ERROR("matrices couldn't be multiplied");
  }
  if ((answ.rows != left.rows) || (answ.cols != right.cols)) {
    ERROR("reinit");
  }
  for (int j = 0; j < answ.cols; j++) {
    for (int i = 0; i < answ.rows; i++) {
      answ.e[i + j * answ.m] = 0.0;
      for (int k = 0; k < left.cols; k ++){
        answ.e[i + j * answ.m] += left.e[i + k * left.m] * right.e[k + j * right.m];

      }
    }
  }
}

__global__
void iden_matr(d_matrix matrix) {
  register int i = threadIdx.x;
  register int j = blockIdx.x;
  register int box = i + j * matrix.m;
  if (i != j) {
    matrix.e[box] = 0.0;
  } else {
    matrix.e[box] = 1.0;
  }
}

__global__
void copyMatrix(d_matrix left, d_matrix right) {
  int l_num = threadIdx.x + blockIdx.x * left.m;
  int r_num = threadIdx.x + blockIdx.x * right.m;
  left.e[l_num] = right.e[r_num];
}

__global__
void multip(d_matrix left, d_matrix right, d_matrix answ) {
  int row = threadIdx.x;
  int col = threadIdx.y;

  int block_rows = blockDim.x;
  int block_cols = blockDim.y;

  int box = row + col * block_rows;

  __shared__ double left_box[BLOCK_SIZE * BLOCK_SIZE + 1];
  __shared__ double right_box[BLOCK_SIZE * BLOCK_SIZE + 1];

  double val = 0.0;

  int bound = gridDim.y;
  int a_col = col, b_row = row;
  int a_row = row + blockIdx.x * block_rows, b_col = col + blockIdx.y * block_cols;
  for (int m = 0; m < bound; m++) {
    left_box[box] = left.e[a_row + a_col * left.m];
    right_box[box] = right.e[b_row + b_col * right.m];

    __syncthreads();

    for (int e = 0; e < block_rows; e++) {
      val += left_box[row + e * block_rows] * right_box[e + col * block_rows];
    }
    a_col += block_cols;
    b_row += block_rows;
    __syncthreads();
  }
  answ.e[a_row + b_col * answ.m] = val;
}

