/*
 * CudaTime.h
 *
 *  Created on: 15 июня 2017 г.
 *      Author: valerius
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
