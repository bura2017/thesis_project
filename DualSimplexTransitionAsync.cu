#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static double *dev_row; // for pivot row
static int threads;
static int flag;
static double *piv_box;
static double *piv_row;
static cudaStream_t str_tr_ma;
static data_async data0, data1;

static void memInit(const int rows, const int cols, int m) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  threads = (rows < prop.maxThreadsPerBlock ? rows : prop.maxThreadsPerBlock);
  int size = MAX_BLOCKS * m;

  data0.dev_matrix.rows = rows;
  data0.dev_matrix.cols = MAX_BLOCKS;
  data0.dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_matrix.e, sizeof(double) * size));

  data1.dev_matrix.rows = rows;
  data1.dev_matrix.cols = MAX_BLOCKS;
  data1.dev_matrix.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_matrix.e, sizeof(double) * size));

  CHECK_CUDA (cudaMalloc ((void**)&data0.dev_col, sizeof(double) * m));
  CHECK_CUDA (cudaMalloc ((void**)&data1.dev_col, sizeof(double) * m));
  CHECK_CUDA (cudaMalloc ((void**)&dev_row, sizeof(double) * cols));

  CHECK_CUDA(cudaStreamCreate(&data0.stream));
  CHECK_CUDA(cudaStreamCreate(&data1.stream));

  data0.pin_matrix = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
  data1.pin_matrix = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
}
static void memFree () {
  cudaFree(data0.dev_matrix.e);
  cudaFree(data1.dev_matrix.e);

  cudaFree(data0.dev_col);
  cudaFree(data1.dev_col);
  cudaFree(dev_row);

  CHECK_CUDA(cudaStreamDestroy(data0.stream));
  CHECK_CUDA(cudaStreamDestroy(data1.stream));

  cudaFreeHost (piv_box);
  cudaFreeHost (piv_row);


  data0.pin_matrix->freeHost();
  data1.pin_matrix->freeHost();
  delete data0.pin_matrix;
  delete data1.pin_matrix;
  //std::cout << flag << std::endl;
}
void transFree(d_matrix &temp_trans_1, d_matrix &temp_trans_2, d_matrix &right_temp) {
  cudaFree(temp_trans_1.e);
  cudaFree(temp_trans_2.e);
  cudaFree(right_temp.e);
  CHECK_CUDA(cudaStreamDestroy(str_tr_ma));
}

int gpuDualSimplexAsync (Matrix &matrix, d_matrix &dev_trans) {
  CHECK_NULL(matrix.e);
  CHECK_NULL(dev_trans.e);

  //std::cout << "First simplex method...";
  memInit(matrix.rows, matrix.cols, matrix.m);
  d_matrix temp_trans_1, temp_trans_2, right_temp;
  dev_trans_init(temp_trans_1, matrix.cols);
  dev_trans_init(temp_trans_2, matrix.cols);
  dev_trans_init(right_temp, matrix.cols);

  CHECK_CUDA(cudaStreamCreate(&str_tr_ma));

  while (1) {
    flag ++;
    if (flag % 1000000 == 0) {
      memFree ();
      transFree(temp_trans_1, temp_trans_2, right_temp);
      return 0;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      cudaDeviceSynchronize();
      double *temp_addr = dev_trans.e;
      if (flag % 2) {
        dev_trans.e = temp_trans_1.e;
        temp_trans_1.e = temp_addr;
      } else {
        dev_trans.e = temp_trans_2.e;
        temp_trans_2.e = temp_addr;
      }
      memFree ();
      transFree(temp_trans_1, temp_trans_2, right_temp);
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      memFree ();
      transFree(temp_trans_1, temp_trans_2, right_temp);
      return 0;
    }
    //std::cout << flag << ' ' << pivot_row << ' ' << pivot_column << std::endl;

    piv_box[0] = - 1 / matrix.e[pivot_row + pivot_col * matrix.m];
    for (int i = 0; i < matrix.cols; i++) {
      piv_row[i] = matrix.e[pivot_row + i * matrix.m];
    }

    int side = right_temp.m;
    iden_matr<<<side,side>>> (right_temp);
    CHECK_CUDA(cudaMemcpy(&(right_temp.e[pivot_col + pivot_col * right_temp.m]), &(piv_box[0]),
        sizeof(double) * 1, cudaMemcpyHostToDevice));
    side = right_temp.cols;
    CHECK_CUDA (cudaMemcpy (dev_row, piv_row, sizeof (double) * side, cudaMemcpyHostToDevice));
    fill_right_trans <<<1,side>>>(right_temp, pivot_col, dev_row);

    modifyTransMatrAsync(flag, pivot_row, pivot_col, temp_trans_1, temp_trans_2, right_temp, str_tr_ma);
    matrixTransformAsync (matrix, pivot_row, pivot_col, threads, data0, data1);
  }
}
