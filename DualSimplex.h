/*
 * DualSimplex.h
 *
 *  Created on: 15 февр. 2017 г.
 *      Author: valerius
 */

#ifndef DUALSIMPLEX_H_
#define DUALSIMPLEX_H_

#include "HandleError.h"
#include "Matrix.h"
#include "MatrixMultip.h"
#include "PivotRowColumn.h"
#include "MatrixTransformation.h"
#include "TransitionMatrix.h"

int cpuDualSimplex (Matrix &matrix);
int gpuDualSimplexAsync (Matrix &matrix);
int gpuDualSimplexSync (Matrix &matrix);

int *gpuDualSimplexDouble (Matrix &matrix0, Matrix &matrix1);

#endif /* DUALSIMPLEX_H_ */
