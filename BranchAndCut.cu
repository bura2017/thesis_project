/**
 * Copyright (c) 2016 ISP RAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CublasMultip.h"
#include "BranchAndCut.h"
#include <fstream>

int num_of_probs;
static d_matrix dev_trans;

static void memInit(int cols) {
  dev_trans.rows = cols;
  dev_trans.cols = cols;
  dev_trans.m = cols + BLOCK_SIZE - 1;
  CHECK_CUDA(cudaMalloc(&dev_trans.e, sizeof(double) * dev_trans.m * dev_trans.m));
  iden_matr<<<dev_trans.m, dev_trans.m>>>(dev_trans);
}

static void memFree(Matrix &matrix) {
  std::cout << num_of_probs << " probs "<< std::endl;
  matrix.freeHost();
  cudaFree(dev_trans.e);
}

void MatMul(const Matrix &cuts, Matrix &result) {
  int side = (cuts.rows > BLOCK_SIZE ? BLOCK_SIZE : cuts.rows);
  dim3 dimBlock(side, side);

  d_matrix temp;
  temp.rows = cuts.rows;
  temp.cols = cuts.cols;
  temp.m = cuts.rows;
  size_t size = sizeof(double) * cuts.rows * cuts.cols;
  CHECK_CUDA(cudaMalloc(&temp.e, size));
  CHECK_CUDA(cudaMemcpy(temp.e, cuts.e, size, cudaMemcpyHostToDevice));

  d_matrix d_cuts;
  d_cuts.rows = (cuts.rows - 1) / side + 1;
  d_cuts.cols = (cuts.cols - 1) / side + 1;
  dim3 dimGrid(d_cuts.rows, d_cuts.cols);
  d_cuts.cols *= side;
  d_cuts.rows *= side;
  d_cuts.m = cuts.rows;
  size = sizeof(double) * d_cuts.rows * d_cuts.cols;
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
  size = sizeof(double) * cuts.rows * cuts.cols;
  cudaMemcpy(result.e, temp.e, size, cudaMemcpyDeviceToHost);

  cudaFree(d_res.e);
  cudaFree(d_cuts.e);
  cudaFree(temp.e);
}


void initMatrix(Matrix &matrix, const Matrix &input, taskTree * &task) {
  //std::cout << "Init new matrix..." << std::endl;
  CHECK_NULL(task);
  matrix = input;

  int cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    cut_rows++;
  }
  if (!cut_rows) {
    CHECK_NULL(NULL);
  }
  Matrix cuts (cut_rows, input.cols);
  cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    int point = branch->prev->point;
    double value = branch->prev->value;
    if (branch->num == 0) {
      cuts.e[cut_rows + point * cuts.m] = 1.0;
      cuts.e[cut_rows + 0 * cuts.m] = value;
    } else {
      cuts.e[cut_rows + point * cuts.m] = -1.0;
      cuts.e[cut_rows + 0 * cuts.m] = -(value + 1.0);
    }
    cut_rows++;
  }
  Matrix temp_matrix(cuts.rows, cuts.cols);
  //Matrix check_matrix(cuts.rows, cuts.cols);
  //MatMul(cuts, temp_matrix);                   //35356.4
  cublas_multip(cuts, dev_trans, temp_matrix); //36686.2
  //std::cout << "Check ... " << cuts.rows << std::endl;
  //check_matrix.print("Check.txt");
  /*for (int i = 0; i < temp_matrix.rows * temp_matrix.cols; i++) {
    if (cmp(temp_matrix.e[i], check_matrix.e[i]) != 0) {
      std::cout << i << ' ' << temp_matrix.e[i] << ' ' << check_matrix.e[i] << std::endl;
    }
  }/**/
  matrix.add_cuts(temp_matrix);
}

bool branchAndCut (Matrix &input) {
  num_of_probs = 0;

  memInit(input.cols);
  if (gpuDualSimplexAsync (input, dev_trans) == 0) {
    std::cout << num_of_probs << std::endl;
    return false;
  }

  taskTree *root = new taskTree;
  orderList *start_order = new orderList (root);

  if (root->branchPoint(input)) {
    std::cout << num_of_probs << std::endl;
    return true;
  }

  Matrix matrix (input, cudaHostAllocDefault, input.cols);

  while (1) {
    num_of_probs++;
    if (num_of_probs % 10000 == 0) {
      std::cout << "bz" << std::endl;
    }

    start_order->task->branchTask();

    for (int l = 0; l < NUM_OF_DAUGHT; l++) {
      initMatrix (matrix, input, start_order->task->next[l]);
      if (gpuDualSimplexAsync (matrix) == 0) {
        delete start_order->task->next[l];
      } else {
        if (start_order->task->next[l]->branchPoint (matrix)) {
          memFree(matrix);
          return true;
        }
        if (start_order->next == NULL) {
          start_order->next = new orderList (start_order->task->next[l]);
        } else {
          start_order->next = start_order->next->pasteTask(start_order->task->next[l]);
        }
      }
    }
    orderList *temp = start_order;
    start_order = start_order->next;
    delete temp;
    if (start_order == NULL) {
      memFree(matrix);
      return false;
    }

  }
}
