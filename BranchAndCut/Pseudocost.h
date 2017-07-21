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

#ifndef PSEUDOCOSTBRANCHING_H_
#define PSEUDOCOSTBRANCHING_H_

struct pseudocost {
private:
  int elems;
  double *l_gains;//objective gains per unit change in variable x_j
  int *l_num;
  double *r_gains;
  int *r_num;

public:
  pseudocost(int elems);
  ~pseudocost();
  double score(int elem, double val);
  int update(double gain, double diff, int point, bool left);
};



#endif /* PSEUDOCOSTBRANCHING_H_ */
