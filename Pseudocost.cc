#include "Pseudocost.h"
#include <cmath>

pseudocost::pseudocost(int elems) : elems(elems),l_gains(new float[elems]),l_num(new int[elems]),
    r_gains(new float[elems]),r_num(new int[elems]) {
  for (int i = 0; i < elems; i++) {
    l_gains[i] = 0.0;
    l_num[i] = 0;
    r_gains[i] = 0.0;
    r_num[i] = 0;
  }
}
pseudocost::~pseudocost() {
  delete [] l_gains;
  delete [] l_num;
  delete [] r_gains;
  delete [] r_num;
}
float pseudocost::score(int elem, float val) {
  if ((l_num[elem] == 0) || (r_num[elem] == 0)) {
    return 0.0;
  }
  if (elem < elems) {
    //TODO: check epsilon
    float eps = 1e-6;

    float f_left = val - floor(val);
    float f_right = 1.0 - f_left;
    float x = f_left * l_gains[elem] / l_num[elem];
    float y = f_right * r_gains[elem] / r_num[elem];

    float s = 1.0;
    s = (x > eps ? x : eps);
    s *= (y > eps ? y : eps);

    return s;
  }
  return 0.0;
}

int pseudocost::update (float gain, float diff, int point, bool left) {
  if (left) {
    l_gains[point] += gain / diff;
    l_num[point] ++;
  } else {
    r_gains[point] += gain / diff;
    r_num[point] ++;
  }

  return 0;
}

