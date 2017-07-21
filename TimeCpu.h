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
