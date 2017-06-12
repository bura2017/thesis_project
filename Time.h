/*
 * Time.h
 *
 *  Created on: 28 февр. 2017 г.
 *      Author: valerius
 */

#ifndef TIME_H_
#define TIME_H_

#include <sys/time.h>

double mtime() {
  struct timeval t;

  gettimeofday(&t, NULL);
  double mt = (double)t.tv_sec * 1000.0 + (double)t.tv_usec / 1000.0;

  return mt;
}

#endif /* TIME_H_ */
