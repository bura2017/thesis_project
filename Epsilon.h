/*
 * Epsilon.h
 *
 *  Created on: 16 июня 2017 г.
 *      Author: valerius
 */

#ifndef EPSILON_H_
#define EPSILON_H_

#include <iostream>

#define EPSILON 1e-10

inline int cmp(double x, double y) {
  if (x > y + EPSILON) {
    return 1;
  }
  if (x < y - EPSILON) {
    return -1;
  }
  return 0;
}


#endif /* EPSILON_H_ */
