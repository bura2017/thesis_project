#include "MatrixTransformation.h"
#include <iostream>

static bool getSizes (const int prev, const int col, const int max_col, int &cur, int &blocks) {
  if (prev == max_col) {
    cur = 0;
    blocks = 0;
    return true;
  }
  cur = prev;
  blocks = MAX_BLOCKS;
  if (cur < col) {
    if (cur + blocks > col) {
      blocks = col - cur;
    }
    return false;
  }
  if (cur == col) {
    cur++;
  }
  if (cur > col) {
    if (cur + blocks > max_col) {
      blocks = max_col - cur;
    }
    return false;
  }
  return false;
}
int matrixTransformAsync(Matrix &matrix, const int row, const int col, const int threads, data_async &data0, data_async &data1) {
  double div = - matrix.e[row + col * matrix.m];

  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }

  CHECK_CUDA (cudaMemcpy (data0.dev_col, &(matrix.e[0 + col * matrix.m]), sizeof(double) * matrix.rows,
      cudaMemcpyHostToDevice));
  CHECK_CUDA (cudaMemcpy (data1.dev_col, &(matrix.e[0 + col * matrix.m]), sizeof(double) * matrix.rows,
      cudaMemcpyHostToDevice));

  int innokentiy[] = {0, 0};
  int blocks[] = {0, 0};

  while (true) {
    if (getSizes (innokentiy[1] + blocks[1], col, matrix.cols, innokentiy[0], blocks[0])) {
      break;
    }
    //std::cout << "0: " << innokentiy[0] << ' ' << blocks[0] << std::endl;
    if (getSizes (innokentiy[0] + blocks[0], col, matrix.cols, innokentiy[1], blocks[1])) {
      break;
    }
    //std::cout << "1: " << innokentiy [1] << ' ' << blocks[1] << std::endl;
    CHECK_NULL(memcpy (data0.pin_matrix->e, &(matrix.e[0 + innokentiy[0] * matrix.m]),
        sizeof(double) * blocks[0] * matrix.m));
    CHECK_NULL(memcpy (data1.pin_matrix->e, &(matrix.e[0 + innokentiy[1] * matrix.m]),
        sizeof(double) * blocks[1] * matrix.m));

    CHECK_CUDA(cudaMemcpyAsync(data0.dev_matrix.e, data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyHostToDevice, data0.stream));
    CHECK_CUDA(cudaMemcpyAsync(data1.dev_matrix.e, data1.pin_matrix->e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyHostToDevice, data1.stream));

    //std::cout << threads << std::endl;
    matrixTransform<<<MAX_BLOCKS,threads,0,data0.stream>>>(data0.dev_matrix, row, data0.dev_col);
    matrixTransform<<<MAX_BLOCKS,threads,0,data1.stream>>>(data1.dev_matrix, row, data1.dev_col);

    CHECK_CUDA(cudaMemcpyAsync(data0.pin_matrix->e, data0.dev_matrix.e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyDeviceToHost, data0.stream));
    CHECK_CUDA(cudaMemcpyAsync(data1.pin_matrix->e, data1.dev_matrix.e,
        sizeof(double) * blocks[1] * data1.dev_matrix.m, cudaMemcpyDeviceToHost, data1.stream));


    CHECK_CUDA(cudaStreamSynchronize(data0.stream));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[0] * matrix.m]), data0.pin_matrix->e,
        sizeof(double) * blocks[0] * matrix.m));

    CHECK_CUDA(cudaStreamSynchronize(data1.stream));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[1] * matrix.m]), data1.pin_matrix->e,
        sizeof(double) * blocks[1] * matrix.m));

  }
  if (innokentiy[0]) {
    CHECK_NULL(memcpy (data0.pin_matrix->e, &(matrix.e[0 + innokentiy[0] * matrix.m]),
        sizeof(double) * blocks[0] * matrix.m));
    CHECK_CUDA(cudaMemcpyAsync(data0.dev_matrix.e, data0.pin_matrix->e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyHostToDevice, data0.stream));
    matrixTransform<<<MAX_BLOCKS,threads,0,data0.stream>>>(data0.dev_matrix, row, data0.dev_col);
    CHECK_CUDA(cudaMemcpyAsync(data0.pin_matrix->e, data0.dev_matrix.e,
        sizeof(double) * blocks[0] * data0.dev_matrix.m, cudaMemcpyDeviceToHost, data0.stream));
    CHECK_CUDA(cudaStreamSynchronize(data0.stream));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[0] * matrix.m]), data0.pin_matrix->e,
        sizeof(double) * blocks[0] * matrix.m));
  }

  return 0;
}/**/
