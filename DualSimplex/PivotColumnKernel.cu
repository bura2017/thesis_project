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

#include "PivotRowColumn.h"
#include "../Epsilon.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__
void matrixChoosePivot(d_matrix matrix, int *num_cols, int pivot_row, int bound_y) {
  __shared__ double cache[BLOCK_SIZE * BLOCK_SIZE * 4];
  __shared__ double piv_row_el[BLOCK_SIZE * 2];
  __shared__ int piv_cols[BLOCK_SIZE * 2];
  int row_box = threadIdx.x;
  int col_box = threadIdx.y;
  int box_size = blockDim.x;
  int el_box = row_box + col_box * box_size;
  int bound_x = (matrix.rows - 1) / box_size + 1;

  int row = row_box;
  int col = col_box;
  for (int l = 0; l < bound_y; l++) {
    if (row_box == 0) {
      piv_cols[col_box] = num_cols[col];
      piv_row_el[col_box] = matrix.e[pivot_row + piv_cols[col_box] * matrix.m];
    }
    __syncthreads();
    for (int k = 0; k < bound_x; k++) {
      cache[el_box] = matrix.e[row + piv_cols[col_box] * matrix.m];
      cache[el_box] /= - piv_row_el[col_box];
      __syncthreads();
      //==============================================
      if ((piv_cols[col_box] != 0) && (piv_cols[row_box] != 0)) {
        for (int i = 0; i < box_size; i++) {
          if (cache[i + col_box * box_size] < cache[i + row_box * box_size] - EPSILON) {
            piv_cols[row_box] = 0;
            break;
          }
          if (cache[i + col_box * box_size] > cache[i + row_box * box_size] + EPSILON) {
            piv_cols[col_box] = 0;
            break;
          }
        }
      }
      //===============================================
      row += box_size;
    }
    __syncthreads();
    if (el_box == 0) {
      for (int n = 0; n < box_size; n++) {
        if (piv_cols[n] != 0) {
          piv_cols[box_size - 1] = piv_cols[n];
          break;
        }
      }
    }
    __syncthreads();

    col += box_size - 1;
  }
  if (el_box == 0) {
    num_cols[0] = piv_cols[box_size - 1];
  }
  __syncthreads();

}
