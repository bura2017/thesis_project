/*
 * CublasMultip.h
 *
 *  Created on: 03 июня 2017 г.
 *      Author: valerius
 */

#ifndef CUBLASMULTIP_H_
#define CUBLASMULTIP_H_

#include "Matrix.h"

void cublas_multip (Matrix const &left, Matrix const &right, Matrix &answ);
void cublas_multip (Matrix const &left, d_matrix const &right, Matrix &answ);
void cublas_multip (d_matrix &left, d_matrix &right, d_matrix &answ, cudaStream_t stream);
void cublas_multip (d_matrix &dev_left, d_matrix &dev_right, d_matrix &dev_answ);

void multip(Matrix const &left, Matrix const &right, Matrix &answ);

__global__ void multip(d_matrix left, d_matrix right, d_matrix answ);
int MatMul(const Matrix &cuts, const d_matrix dev_trans, Matrix &result);
#endif /* CUBLASMULTIP_H_ */
