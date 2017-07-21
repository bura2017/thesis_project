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

#include "Matrix.h"
#include "HandleError.h"
#include "BranchAndCut/BranchAndCut.h"
#include "CudaDeviceProperties.h"
#include "TimeCuda.h"
#include "GenTest.h"
#include "Epsilon.h"

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <fstream>
#include <unistd.h>

int flag;

static int testingSimplex (int vars, int ineqs, int test_num, double &cpu_time, double &dev_time, double &bac_time) {
  char filename[MAX_LENG];
  sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

  char fullname[MAX_LENG];
  sprintf(fullname, "TestGenerator/%s", filename);
  if (access(fullname, 0) == -1) {
    gen_test(test_num, vars, ineqs, flag);
  }

  Matrix input(fullname);

  cuda_time time;
  cmp ((double) input.cols, (double) input.rows);

  int iters_man = 0;

  {
    time.start();
    Matrix matrix(input);
    int iters_cpu = cpuDualSimplex (matrix);
    time.stop();
    iters_man = iters_cpu;
    if (iters_man == 0) {
      return 1;
    }
    if (iters_cpu != iters_man) {
      std::cout << iters_cpu << " != " << iters_man << std::endl;
      std::cout << "ERROR wrong answer cpu" << std::endl;
    }
    if (checkCorrect(input, matrix) == 0) {
      std::cout << "ERROR wrong answer " << std::endl;
    }
    cpu_time += time.time();

  }

  {
    int supply = input.rows % (BLOCK_SIZE * 2);
    supply = (supply > 0 ? BLOCK_SIZE * 2 - supply : 0);
    time.start();
    Matrix matrix(input, supply);
    int iters_dev = gpuDualSimplexSyncDev (matrix);
    time.stop();
    if (iters_dev != iters_man) {
      return 1;
      std::cout << iters_dev << " != " << iters_man << std::endl;
      std::cout << "ERROR wrong answer dev sync" << std::endl;
    }
    dev_time += time.time();
  }

  {
    int supply = input.rows % (BLOCK_SIZE * 2);
    supply = (supply > 0 ? BLOCK_SIZE * 2 - supply : 0);
    time.start();
    Matrix matrix(input.supply);
    int iters = branchAndBound(matrix);
    time.stop();
    bac_time += time.time();
  }

  return 0;
}
int main (int argc, char **argv) {
  cudaSetDevice(0);
  cudaDeviceReset();
  CHECK_CUDA (cudaSetDeviceFlags (cudaDeviceMapHost));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties (&prop, 0));

  if (!prop.deviceOverlap) {
    std::cout << "Device will not handle overlaps, so no speed up from streams" << std::endl;
  }
  if (prop.integrated) {
    std::cout << "Integrated device gives speed up from zero-copy memory" << std::endl;
  }

  std::ofstream results_cpu("results_cpu.txt");
  std::ofstream results_dev("results_dev.txt");

  {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    double bac_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    int vars = 512;
    int ineqs = 511;
    while (test_num < 10) {
      if (testingSimplex(vars, ineqs, test_num, cpu_time, dev_time, bac_time)) {
        continue;
      }
      test_num++;
    }
    results_cpu << cpu_time << std::endl;
    results_dev << dev_time << std::endl;
    std::cout << vars + 1 << "x" << ineqs + 1 << std::endl;
    std::cout << "dual simplex result cpu_time " << cpu_time << " gpu_time " << dev_time << std::endl;
    std::cout << "branch-and-cut time " << bac_time << std::endl;
  }

  return 0;

}
