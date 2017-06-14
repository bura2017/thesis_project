#include "TransitionMatrix.h"
#include "MatrixMultip.h"


int dev_trans_init(d_matrix &dev_trans, int side) {
  dev_trans.rows = side;
  dev_trans.cols = side;
  dev_trans.m = side + BLOCK_SIZE - 1;
  CHECK_CUDA(cudaMalloc(&dev_trans.e, sizeof(double) * dev_trans.m * dev_trans.m));
  iden_matr<<<dev_trans.m, dev_trans.m>>>(dev_trans);

  return 0;
}

__global__
void fill_right_trans(d_matrix matrix, int col, double *row) {
  int j = threadIdx.x;
  __shared__ double piv_num;
  if (j == col) {
    piv_num = matrix.e[col + col * matrix.m];
  }
  __syncthreads();

  if (j != col) {
    int n = col + j * matrix.m;
    matrix.e[n] = piv_num * row[j];
  }
}

void modifyTransMatrAsync (int flag, int pivot_row, int pivot_col, d_matrix &temp_trans_1, d_matrix &temp_trans_2,
    d_matrix right_temp, cudaStream_t str_tr_ma) {

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(right_temp.m / blockDim.x, right_temp.m / blockDim.y);
  CHECK_CUDA(cudaStreamSynchronize(str_tr_ma));
  if (flag % 2) {
    cublas_multip (temp_trans_1, right_temp, temp_trans_2, str_tr_ma);
  } else {
    cublas_multip (temp_trans_2, right_temp, temp_trans_1, str_tr_ma);
  }
}
