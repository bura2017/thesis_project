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

#include <iostream>

int branchPoint(Matrix &matrix, int &point, double &value, double &diff_best) {
  point = 0;
  value = 0.0;
  diff_best = 0.0;

  for (int i = 1; i < matrix.cols; i++){
    double val = matrix.e[i + 0 * matrix.m];
    if ((val > 0.0) && (val < 1e+12)) {
      double diff = val - round(val);
      if (diff < 0) {
        diff = - diff;
      }
      if ((cmp(diff,0) == 1) && ((!point) || ((point) && (diff_best < diff)))) {
          diff_best = diff;
          point = i;
          value = floor(val);
      }
    }
  }

  return 0;

}
int branchPoint (Matrix &matrix, int &point, double &value, double &diff_best, pseudocost &cost) {
  double s_best = 0.0;
  point = 0;
  value = 0.0;
  diff_best = 0.0;
  int reliability = 0;

  for (int i = 0; i < matrix.cols; i++) {
    double val = matrix.e[i + 0 * matrix.m];
    if ((val > 0.0) && (val < 1e+12)) {
      double diff = val - round(val);
      if (diff < 0) {
        diff = -diff;
      }
      if ((cmp(diff,0) == 1)) {
        double s = cost.score(i, val);
        if (s_best < s) {
          s_best = s;
          point = i;
          value = floor(val);
          diff_best = diff;
        }
      }
    }
  }
  if (point == 0) {
    return branchPoint(matrix, point, value, diff_best);
  }

  return point;
}
