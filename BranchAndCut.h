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
#include "Pseudocost.h"

#define MAX_NUM_OF_CUTS 100
#define NUM_OF_DAUGHT 2
#define BRANCH_APPROX 1e-10

struct taskTree {
  int point;
  taskTree *prev;
  taskTree *next[NUM_OF_DAUGHT];
  double value;
  int num; //parent's daughter num
  int num_of_int;
  double func;
  double diff;
  double *cuts;

  taskTree (int point, taskTree* prev, double value, int num, int num_of_int = -1,
      double func = 0.0, double diff = 0.0);
  int branchTask(Matrix &matrix, pseudocost *cost);
  int countInts(Matrix &matrix);
  ~taskTree () ;
};

struct orderList {
  taskTree* task;
  orderList* next;

  orderList (taskTree *task, orderList *next = NULL) : task(task), next(next) {}
  orderList *pasteTask (taskTree *task) ;
};

int branchAndBound (Matrix &input);
void initMatrix(Matrix &matrix, const Matrix &input, taskTree * &task, d_matrix dev_trans);

int branchPoint(Matrix &matrix, int &point, double &val, double &diff_best);
int branchPoint (Matrix &matrix, int &point, double &value, double &diff_best, pseudocost &cost);

int mirCuts (Matrix &matrix, double *ineq);

#endif /* BRANCHANDCUT_H_ */
