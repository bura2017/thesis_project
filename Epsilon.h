/*
 * Epsilon.h
 *
 *  Created on: 16 июня 2017 г.
 *      Author: valerius
 */

#ifndef EPSILON_H_
#define EPSILON_H_

#include <iostream>

inline int cmp(double x, double y) {
  static double epsilon = pow (10, - 20 * log10(x * y));
  if (x > y + epsilon) {
    return 1;
  }
  if (x < y - epsilon) {
    return -1;
  }
  return 0;
}


#endif /* EPSILON_H_ */
