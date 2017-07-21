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

#include "BranchAndCut.h"
#include "TransitionMatrix.h"

#include <fstream>

static int num_of_probs;

static void memFree(d_matrix transition) {
  std::cout << "branch-and-cut result " << (num_of_probs > 0 ? "sat" : "unsat") << ' ' << num_of_probs << " probs " << std::endl;
  dev_trans_free(transition);
}

int branchAndBound (Matrix &input) {
  std::cout << "Start branch and bound ..." << std::endl;
  num_of_probs = 1;

  if (gpuDualSimplexSyncDev (input) < 0) {
    std::cout << "first unsat" << std::endl;
    return -num_of_probs;
  }
  d_matrix transition;
  dev_trans_init(transition, input);

  taskTree *root = new taskTree(0, NULL, 0.0, 0);
  orderList *start_order = new orderList (root);

  if (root->branchTask(input, NULL) == 0) {
    std::cout << "first sat" << std::endl;
    return num_of_probs;
  }

  Matrix matrix0 (input, input.cols + MAX_NUM_OF_CUTS);
  Matrix matrix1 (input, input.cols + MAX_NUM_OF_CUTS);

  pseudocost cost(input.cols);
  int cost_check = (input.cols / 4 < 10 ? 10 : input.cols / 4);
  pseudocost *cost_rel = NULL;

  while (1) {
    num_of_probs++;
    if (num_of_probs == cost_check) {
      cost_rel = &cost;
    }

    taskTree *next_0 = start_order->task->next[0];
    taskTree *next_1 = start_order->task->next[1];
    initMatrix (matrix0, input, next_0, transition);
    initMatrix (matrix1, input, next_1, transition);
    int *result = gpuDualSimplexDouble(matrix0, matrix1);

    //========ADD CUTS================
    int add_cut_0 = 0, add_cut_1 = 0;
    if (result[0] > 0) {
      add_cut_0 = mirCuts(matrix0, next_0->cuts);
    }
    if (result[1] > 0) {
      add_cut_1 = mirCuts(matrix1, next_1->cuts);
    }
    if ((add_cut_0 > 0) && (add_cut_1 > 0)) {
      initMatrix (matrix0, input, next_0, transition);
      initMatrix (matrix1, input, next_1, transition);
      delete [] result;
      result = gpuDualSimplexDouble(matrix0, matrix1);
    }
    if ((add_cut_0 > 0) && (add_cut_1 == 0)) {
      initMatrix (matrix0, input, next_0, transition);
      result[0] = gpuDualSimplexAsync(matrix0);
    }
    if ((add_cut_0 == 0) && (add_cut_1 > 0)) {
      initMatrix (matrix1, input, next_1, transition);
      result[1] = gpuDualSimplexAsync(matrix1);
    }

    //=======UPDATE PSEUDOCOST=========
    if (result[0] > 0) {
      cost.update(next_0->prev->func - matrix0.e[0], next_0->diff, next_0->point, true);
    }
    if (result[1] > 0) {
      cost.update(next_1->prev->func - matrix1.e[0], next_1->diff, next_1->point, false);
    }
    //==========BRANCH=================
    int b = -1;
    while (b < 0) {
      if (result[0] < 0) {
        delete next_0;
        break;
      }
      b = next_0->branchTask (matrix0, cost_rel);
      if (b == 0) {
        memFree(transition);
        return num_of_probs;
      }
      if (b < 0) {
        initMatrix (matrix0, input, next_0, transition);
        result[0] = gpuDualSimplexAsync(matrix0);
        if (result[0] > 0) {
          cost.update(next_0->prev->func - matrix0.e[0], next_0->diff, next_0->point, true);
        }
      }
      if (b > 0) {
        if (start_order->next == NULL) {
          start_order->next = new orderList (next_0);
        } else {
          start_order->next = start_order->next->pasteTask(next_0);
        }
      }
    }
    b = -1;
    while (b < 0) {
      if (result[1] < 0) {
        delete next_1;
        break;
      }
      b = next_1->branchTask (matrix1, cost_rel);
      if (b == 0) {
        memFree(transition);
        return num_of_probs;
      }
      if (b < 0) {
        initMatrix (matrix1, input, next_1, transition);
        result[1] = gpuDualSimplexAsync(matrix1);
        if (result[1] > 0) {
          cost.update(next_1->prev->func - matrix1.e[0], next_1->diff, next_1->point, true);
        }
      }
      if (b > 0) {
        if (start_order->next == NULL) {
          start_order->next = new orderList (next_1);
        } else {
          start_order->next = start_order->next->pasteTask(next_1);
        }
      }
    }

    delete [] result;

    orderList *temp = start_order;
    start_order = start_order->next;
    delete temp;
    if (start_order == NULL) {
      memFree(transition);
      return -num_of_probs;
    }

  }
}
