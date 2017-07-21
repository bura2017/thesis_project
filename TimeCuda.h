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

#ifndef CUDATIME_H_
#define CUDATIME_H_

struct cuda_time {
private:
  cudaEvent_t _start;
  cudaEvent_t _stop;
  float _ms;
public:
  cuda_time() {
    CHECK_CUDA(cudaEventCreate(&_start));
    CHECK_CUDA(cudaEventCreate(&_stop));
    _ms = 0;

  }
  ~cuda_time() {
    CHECK_CUDA(cudaEventDestroy(_start));
    CHECK_CUDA(cudaEventDestroy(_stop));
  }
  void start() {
    CHECK_CUDA(cudaEventRecord(_start, 0));
  }
  void stop() {
    CHECK_CUDA(cudaEventRecord(_stop, 0));
    CHECK_CUDA(cudaEventSynchronize(_stop));
    CHECK_CUDA(cudaEventElapsedTime(&_ms, _start, _stop));
  }
  double time () {
    return _ms;
  }
};



#endif /* CUDATIME_H_ */
