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

#ifndef MATRIXTRANSITION_H_
#define MATRIXTRANSITION_H_

#include "../Matrix.h"

int dev_trans_init(d_matrix &dev_trans, Matrix &input);
int dev_trans_free(d_matrix &dev_trans);
__global__ void fill_right_trans(d_matrix matrix, int col, double *row);
void modifyTransMatrAsync (int flag, int pivot_row, int pivot_col, d_matrix &temp_trans_1, d_matrix &temp_trans_2,
    d_matrix right_temp, cudaStream_t str_tr_ma);

#endif /* MATRIXTRANSITION_H_ */
