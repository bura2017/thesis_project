#include "BranchAndCut.h"
#include "Epsilon.h"

  taskTree::taskTree (int point, taskTree* prev, float value, int num, int num_of_int, float func, float diff) :
    point(point), prev(prev), value(value), num(num), num_of_int(num_of_int) , func(func), diff(diff), cuts(NULL) {
    for (int i = 0; i < NUM_OF_DAUGHT; i++) {
      next[i] = NULL;
    }
  }
  int taskTree::branchTask(Matrix &matrix, pseudocost *cost) {
    //std::cout << "======branch task=======" << std::endl;
    int branch_point = 0;
    float val = 0;
    func = matrix.e[0];

    if (cost == NULL) {
      branchPoint(matrix, branch_point, val, diff);
    } else {
      branchPoint(matrix, branch_point, val, diff, *cost);
    }
    if (branch_point == 0) {
      return 0;
    }
    for (taskTree *t = this; t != NULL; t = t->prev) {
      if (branch_point == t->point) {
        if (t->num == 0) {
          if (cmp(val, t->value) == -1) {
            t->value = val;
            return -1;
          }
        }
        if (t->num == 1) {
          if (cmp(val + 1.0, t->value) == 1) {
            t->value = val + 1.0;
            return -1;
          }
        }
        matrix.e[branch_point] = -1.1;
        return branchTask(matrix, cost);
      }
    }

    countInts(matrix);
    next[0] = new taskTree(branch_point, this, val, 0);
    next[1] = new taskTree(branch_point, this, val + 1.0, 1);

    return 1;
  }

  int taskTree::countInts(Matrix &matrix) {
    num_of_int = 0;
    for (int i = 0; i < matrix.cols; i++) {
      float val = matrix.e[i];
      float diff = val - round(val);
      if (diff < 0.0) {
        diff = - diff;
      }
      if (cmp(diff,0) == 0) {
        num_of_int++;
      }
    }
    return 0;
  }

  taskTree::~taskTree () {
    if (prev!=NULL) {
      prev->next[num] = NULL;
    }
    if (cuts != NULL) {
      delete [] cuts;
    }
  }
