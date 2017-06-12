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
#include "CublasMultip.h"

int cpuDualSimplex (Matrix &matrix);
int gpuDualSimplexAsync (Matrix &matrix);
int gpuDualSimplexSync (Matrix &matrix);
int gpuDualSimplexCublas (Matrix &matrix);
int gpuDualSimplexSyncDev (Matrix &matrix);

int cpuDualSimplex (Matrix &matrix, Matrix &transition);
int gpuDualSimplexAsync (Matrix &matrix, d_matrix &dev_trans);

#endif /* DUALSIMPLEX_H_ */
