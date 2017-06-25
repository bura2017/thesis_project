#include "MatrixMultip.h"

__global__
void multip(d_matrix left, d_matrix right, d_matrix answ) {
  int row = threadIdx.x;
  int col = threadIdx.y;

  int block_rows = blockDim.x;
  int block_cols = blockDim.y;

  int box = row + col * block_rows;

  __shared__ float left_box[BLOCK_SIZE * BLOCK_SIZE + 1];
  __shared__ float right_box[BLOCK_SIZE * BLOCK_SIZE + 1];

  float val = 0.0;

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

int MatMul(const Matrix &cuts, const d_matrix dev_trans, Matrix &result) {
  int side = (cuts.rows > BLOCK_SIZE ? BLOCK_SIZE : cuts.rows);
  dim3 dimBlock(side, side);

  d_matrix temp;
  temp.rows = cuts.rows;
  temp.cols = cuts.cols;
  temp.m = cuts.rows;
  size_t size = sizeof(float) * cuts.rows * cuts.cols;
  CHECK_CUDA(cudaMalloc(&temp.e, size));
  CHECK_CUDA(cudaMemcpy(temp.e, cuts.e, size, cudaMemcpyHostToDevice));

  d_matrix d_cuts;
  d_cuts.rows = (cuts.rows - 1) / side + 1;
  d_cuts.cols = (cuts.cols - 1) / side + 1;
  dim3 dimGrid(d_cuts.rows, d_cuts.cols);
  d_cuts.cols *= side;
  d_cuts.rows *= side;
  d_cuts.m = cuts.rows;
  size = sizeof(float) * d_cuts.rows * d_cuts.cols;
  cudaMalloc(&d_cuts.e, size);
  iden_matr<<<d_cuts.cols,d_cuts.rows>>> (d_cuts);
  copyMatrix<<<temp.cols,temp.rows>>>(d_cuts,temp);
  d_cuts.rows = cuts.rows;
  d_cuts.cols = cuts.cols;

  d_matrix d_res;
  d_res.rows = d_cuts.rows;
  d_res.cols = d_cuts.cols;
  d_res.m = d_cuts.m;
  cudaMalloc(&d_res.e, size);

  multip<<<dimGrid, dimBlock>>>(d_cuts, dev_trans, d_res);

  copyMatrix<<<cuts.cols,cuts.rows>>>(temp, d_res);
  size = sizeof(float) * cuts.rows * cuts.cols;
  cudaMemcpy(result.e, temp.e, size, cudaMemcpyDeviceToHost);

  cudaFree(d_res.e);
  cudaFree(d_cuts.e);
  cudaFree(temp.e);

  return 0;
}
