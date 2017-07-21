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

#ifndef DUALSIMPLEX_H_
#define DUALSIMPLEX_H_

#include "../HandleError.h"
#include "../Matrix.h"
#include "../MatrixMultip/MatrixMultip.h"
#include "PivotRowColumn.h"
#include "../MatrixTransform/MatrixTransformation.h"

int cpuDualSimplex (Matrix &matrix);
int gpuDualSimplexAsync (Matrix &matrix);
int gpuDualSimplexSync (Matrix &matrix);
int gpuDualSimplexSyncDev (Matrix &matrix);

int *gpuDualSimplexDouble (Matrix &matrix0, Matrix &matrix1);

#endif /* DUALSIMPLEX_H_ */
