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

#ifndef BRANCHANDCUT_H_
#define BRANCHANDCUT_H_

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <iostream>

#include "DualSimplex.h"
#include "Matrix.h"

#define MAX_NUM_OF_CUTS 500
#define NUM_OF_DAUGHT 2

struct taskTree {
  int point;
  taskTree *prev;
  taskTree *next[NUM_OF_DAUGHT];
  double value;
  int num; //parent's daughter num
  int num_of_int;

  taskTree (int point = 0, taskTree* prev = NULL, double value = 0, int num = 0, int num_of_int = -1) :
    point(point), prev(prev), value(value), num(num), num_of_int(num_of_int) {
    for (int i = 0; i < NUM_OF_DAUGHT; i++) {
      next[i] = NULL;
    }
  }
  void branchTask() {
    for (int i = 0; i < NUM_OF_DAUGHT; i++) {
      next[i] = new taskTree(0,this,0,i,-1);
    }
  }
  int branchPoint(Matrix const &matrix) {
    int branch_point = 0;
    double val = 0.0;
    int ints = 0;

    int *fix_vals = new int[matrix.cols];
    for (int i = 0; i < matrix.cols; i++) {
      fix_vals[i] = 0;
    }
    for (taskTree *t = this; t != NULL; t = t->prev) {
      fix_vals[t->point] = 1;
    }

    for (int i = 1; i < matrix.cols; i++){
      val = matrix.e[i + 0 * matrix.m];
      int c = cmp(val, round(val));
      if ((c != 0) && (!branch_point) && (fix_vals[i] == 0)) {
        branch_point = i;
        val = floor(val);
      }
      if ((c == 0) || (fix_vals[i] == 1)) {
        ints++;
      }
    }

    if (branch_point == 0) {
      return 1;
    }

    point = branch_point;
    value = val;
    num_of_int = ints;

    delete [] fix_vals;

    return 0;

  }
  ~taskTree () {
    if (prev!=NULL) {
      prev->next[num] = NULL;
    }
  }
};

struct orderList {
  taskTree* task;
  orderList* next;

  orderList (taskTree *task, orderList *next = NULL) : task(task), next(next) {}
  orderList *pasteTask (taskTree *task) {
    orderList *order = new orderList (task);

    if (task->num_of_int > this->task->num_of_int) {
      order->next = this;
      return order;
    }
    orderList *current = this;
    while (current->next != NULL) {
      if (current->next->task->num_of_int < task->num_of_int) {
        break;
      }
      current = current->next;
    }
    order->next = current->next;
    current->next = order;
    return this;
  }
};

bool branchAndCut (Matrix &input);

#endif /* BRANCHANDCUT_H_ */
