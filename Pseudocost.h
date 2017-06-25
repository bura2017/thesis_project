/*
 * PseudocostBranching.h
 *
 *  Created on: 18 июня 2017 г.
 *      Author: valerius
 */

#ifndef PSEUDOCOSTBRANCHING_H_
#define PSEUDOCOSTBRANCHING_H_

struct pseudocost {
private:
  int elems;
  float *l_gains;//objective gains per unit change in variable x_j
  int *l_num;
  float *r_gains;
  int *r_num;

public:
  pseudocost(int elems);
  ~pseudocost();
  float score(int elem, float val);
  int update(float gain, float diff, int point, bool left);
};



#endif /* PSEUDOCOSTBRANCHING_H_ */
