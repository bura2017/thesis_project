#include "DualSimplex.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

static d_matrix dev_matrix0, dev_matrix1;
static Matrix *matrix0, *matrix1;
static double *dev_col0, *dev_col1; //for pivot column
static double *dev_row; // for pivot row
static int threads;
static int flag;
static double *piv_box;
static double *piv_row;
//static cudaDeviceProp prop;
cudaStream_t stream0, stream1;
cudaStream_t str_tr_ma;

static void memInit(const int rows, const int cols, int m) {
  flag = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  threads = (rows < prop.maxThreadsPerBlock ? rows : prop.maxThreadsPerBlock);
  int size = MAX_BLOCKS * m;

  dev_matrix0.rows = rows;
  dev_matrix0.cols = MAX_BLOCKS;
  dev_matrix0.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix0.e, sizeof(double) * size));

  dev_matrix1.rows = rows;
  dev_matrix1.cols = MAX_BLOCKS;
  dev_matrix1.m = m;
  CHECK_CUDA (cudaMalloc ((void**)&dev_matrix1.e, sizeof(double) * size));

  CHECK_CUDA (cudaMalloc ((void**)&dev_col0, sizeof(double) * m));
  CHECK_CUDA (cudaMalloc ((void**)&dev_col1, sizeof(double) * m));
  CHECK_CUDA (cudaMalloc ((void**)&dev_row, sizeof(double) * cols));

  CHECK_CUDA(cudaStreamCreate(&stream0));
  CHECK_CUDA(cudaStreamCreate(&stream1));

  CHECK_CUDA(cudaHostAlloc ((void**)&piv_box, sizeof(double) * 1, cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc((void**)&piv_row, sizeof (double) * cols, cudaHostAllocDefault));

  matrix0 = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
  matrix1 = new Matrix(rows, MAX_BLOCKS, cudaHostAllocDefault, m - rows);
}
static void memFree () {
  cudaFree(dev_matrix0.e);
  cudaFree(dev_matrix1.e);

  cudaFree(dev_col0);
  cudaFree(dev_col1);
  cudaFree(dev_row);

  CHECK_CUDA(cudaStreamDestroy(stream0));
  CHECK_CUDA(cudaStreamDestroy(stream1));

  cudaFreeHost (piv_box);
  cudaFreeHost (piv_row);


  matrix0->freeHost();
  matrix1->freeHost();
  delete matrix0;
  delete matrix1;
  //std::cout << flag << std::endl;
}
static int pivotRow(Matrix const &matrix) {
  for (int i = 1; i < matrix.rows; i++) {
    if (cmp(matrix.e[i + 0 * matrix.m],0) == -1) {
      return i;
    }
  }
  return 0;
}
static int pivotColumn(Matrix const &matrix, const int row) {
  int ret = 0;

  for (int j = 1; j < matrix.cols; j++) {
    if (cmp(matrix.e[row + j * matrix.m],0) == -1) {
      if (ret == 0) {
        ret = j;
      } else {
        for (int i = 0; i < matrix.rows; i++) {
          double val1 = matrix.e[i + ret * matrix.m] * matrix.e[row + j * matrix.m];
          double val2 = matrix.e[i + j * matrix.m] * matrix.e[row + ret * matrix.m];
          int c = cmp(val1,val2);
          if (c == -1) {
            ret = j;
            break;
          }
          if (c == 1) {
            break;
          }
        }
      }
    }
  }
  return ret;
}
__global__
static void simplexMatrix(d_matrix matrix, int piv_row, double *col_e) {
  __shared__ double cache[1025];
  __shared__ double matr[1025];
  __shared__ double piv_row_el;
  int row = threadIdx.x;
  int col = blockIdx.x; //column num
  //one column for one block

  cache[row] = col_e[row];
  matr[row] = matrix.e[row + col * matrix.m];
  if (threadIdx.x == piv_row) {
    piv_row_el = matr[piv_row];
  }
  __syncthreads();
  matr[row] += piv_row_el * cache[row];
  matrix.e[row + col * matrix.m] = matr[row];
}/**/
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
static void dualSimplex(Matrix &matrix, const int row, const int col) {
  double div = - matrix.e[row + col * matrix.m];

  for (int i = 0; i < matrix.rows; i++) {
    matrix.e[i + col * matrix.m] /= div;
  }

  CHECK_CUDA (cudaMemcpy (dev_col0, &(matrix.e[0 + col * matrix.m]), sizeof(double) * matrix.rows,
      cudaMemcpyHostToDevice));
  CHECK_CUDA (cudaMemcpy (dev_col1, &(matrix.e[0 + col * matrix.m]), sizeof(double) * matrix.rows,
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
    CHECK_NULL(memcpy (matrix0->e, &(matrix.e[0 + innokentiy[0] * matrix.m]),
        sizeof(double) * blocks[0] * matrix.m));
    CHECK_NULL(memcpy (matrix1->e, &(matrix.e[0 + innokentiy[1] * matrix.m]),
        sizeof(double) * blocks[1] * matrix.m));

    CHECK_CUDA(cudaMemcpyAsync(dev_matrix0.e, matrix0->e,
        sizeof(double) * blocks[0] * dev_matrix0.m, cudaMemcpyHostToDevice, stream0));
    CHECK_CUDA(cudaMemcpyAsync(dev_matrix1.e, matrix1->e,
        sizeof(double) * blocks[1] * dev_matrix1.m, cudaMemcpyHostToDevice, stream1));

    simplexMatrix<<<blocks[0],threads,0,stream0>>>(dev_matrix0, row, dev_col0);
    simplexMatrix<<<blocks[1],threads,0,stream1>>>(dev_matrix1, row, dev_col1);

    CHECK_CUDA(cudaMemcpyAsync(matrix0->e, dev_matrix0.e,
        sizeof(double) * blocks[0] * dev_matrix0.m, cudaMemcpyDeviceToHost, stream0));
    CHECK_CUDA(cudaMemcpyAsync(matrix1->e, dev_matrix1.e,
        sizeof(double) * blocks[1] * dev_matrix1.m, cudaMemcpyDeviceToHost, stream1));


    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[0] * matrix.m]), matrix0->e,
        sizeof(double) * blocks[0] * matrix.m));

    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[1] * matrix.m]), matrix1->e,
        sizeof(double) * blocks[1] * matrix.m));

  }
  if (innokentiy[0]) {
    CHECK_NULL(memcpy (matrix0->e, &(matrix.e[0 + innokentiy[0] * matrix.m]),
        sizeof(double) * blocks[0] * matrix.m));
    CHECK_CUDA(cudaMemcpyAsync(dev_matrix0.e, matrix0->e,
        sizeof(double) * blocks[0] * dev_matrix0.m, cudaMemcpyHostToDevice, stream0));
    simplexMatrix<<<blocks[0],threads,0,stream0>>>(dev_matrix0, row, dev_col0);
    CHECK_CUDA(cudaMemcpyAsync(matrix0->e, dev_matrix0.e,
        sizeof(double) * blocks[0] * dev_matrix0.m, cudaMemcpyDeviceToHost, stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_NULL(memcpy (&(matrix.e[0 + innokentiy[0] * matrix.m]), matrix0->e,
        sizeof(double) * blocks[0] * matrix.m));
  }

}/**/

int gpuDualSimplexAsync (Matrix &matrix) {
  CHECK_NULL(matrix.e);

  //std::cout << "  simplex method... ";
  memInit(matrix.rows, matrix.cols, matrix.m);

  while (1) {
    flag ++;
    //std::cout << flag << std::endl;
    if (flag % 10000 == 0) {
      std::cout << "ups" << std::endl;
      memFree ();
      return 0;
    }

    int pivot_row = pivotRow (matrix);
    if (!pivot_row) {
      memFree ();
      return flag;
    }

    int pivot_col = pivotColumn (matrix, pivot_row);
    if (!pivot_col) {
      memFree ();
      return 0;
    }
    //std::cout << flag << ' ' << pivot_row << ' ' << pivot_column << std::endl;

    dualSimplex (matrix, pivot_row, pivot_col);
  }
}

//==============================================================================================================

void transInit(int side, d_matrix &temp_trans_1, d_matrix &temp_trans_2, d_matrix &right_temp){
  int x = (side - 1) / BLOCK_SIZE + 1;
  x *= BLOCK_SIZE;
  temp_trans_1.rows = side;
  temp_trans_1.cols = side;
  temp_trans_1.m = x;
  CHECK_CUDA (cudaMalloc ((void**)&temp_trans_1.e, sizeof(double) * x * x));
  iden_matr<<<x, x>>>(temp_trans_1);
  temp_trans_2.rows = side;
  temp_trans_2.cols = side;
  temp_trans_2.m = x;
  CHECK_CUDA (cudaMalloc ((void**)&temp_trans_2.e, sizeof(double) * x * x));
  iden_matr<<<x, x>>>(temp_trans_2);

  right_temp.rows = side;
  right_temp.cols = side;
  right_temp.m = x;
  CHECK_CUDA (cudaMalloc ((void**)&right_temp.e, sizeof(double) * x * x));
  iden_matr<<<x, x>>>(right_temp);

  CHECK_CUDA(cudaStreamCreate(&str_tr_ma));
}
void transFree(d_matrix &temp_trans_1, d_matrix &temp_trans_2, d_matrix &right_temp) {
  cudaFree(temp_trans_1.e);
  cudaFree(temp_trans_2.e);
  cudaFree(right_temp.e);
  CHECK_CUDA(cudaStreamDestroy(str_tr_ma));
}

void modifyTransMatrAsync (int pivot_row, int pivot_col, d_matrix &temp_trans_1, d_matrix &temp_trans_2,
    d_matrix right_temp) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(right_temp.m / blockDim.x, right_temp.m / blockDim.y);
  CHECK_CUDA(cudaStreamSynchronize(str_tr_ma));
  if (flag % 2) {
    cublas_multip (temp_trans_1, right_temp, temp_trans_2, str_tr_ma);
  } else {
    cublas_multip (temp_trans_2, right_temp, temp_trans_1, str_tr_ma);
  }
}/**/
__global__
static void fill_right_trans(d_matrix matrix, int col, double *row) {
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
int gpuDualSimplexAsync (Matrix &matrix, d_matrix &dev_trans) {
  CHECK_NULL(matrix.e);
  CHECK_NULL(dev_trans.e);

  //std::cout << "First simplex method...";
  memInit(matrix.rows, matrix.cols, matrix.m);
  d_matrix temp_trans_1, temp_trans_2, right_temp;
  transInit(matrix.cols, temp_trans_1, temp_trans_2, right_temp);

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
      int side = temp_trans_1.rows;
      if (flag % 2) {
        copyMatrix<<<side,side>>>(dev_trans, temp_trans_1);
      } else {
        copyMatrix<<<side,side>>>(dev_trans, temp_trans_2);
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

    modifyTransMatrAsync(pivot_row, pivot_col, temp_trans_1, temp_trans_2, right_temp);
    dualSimplex (matrix, pivot_row, pivot_col);
  }
}
