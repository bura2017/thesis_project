#include "Matrix.h"
#include "HandleError.h"
#include "BranchAndCut.h"
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

static int testingSimplex (int vars, int ineqs, int test_num, double &cpu_time, double &dev_time) {
  char filename[MAX_LENG];
  sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

  char fullname[MAX_LENG];
  sprintf(fullname, "TestGenerator/%s", filename);
  //if (access(fullname, 0) == -1) {
    gen_test(test_num, vars, ineqs, flag);
  //}

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
    //std::cout << iters_man << std::endl;
    //std::cout << time.time() << std::endl;
    if (iters_cpu != iters_man) {
      std::cout << iters_cpu << " != " << iters_man << std::endl;
      std::cout << "ERROR wrong answer cpu" << std::endl;
    }
    //if (checkCorrect(input, matrix) == 0) {
    //  std::cout << "ERROR wrong answer " << std::endl;
    //}
    cpu_time = time.time();

  }/**/

  {
    int supply = input.rows % (BLOCK_SIZE * 2);
    supply = (supply > 0 ? BLOCK_SIZE * 2 - supply : 0);
    time.start();
    Matrix matrix(input, supply);
    int iters_dev = gpuDualSimplexSyncDev (matrix);
    time.stop();
    //std::cout << time.time() << std::endl;
    if (iters_dev != iters_man) {
      return 1;
      std::cout << iters_dev << " != " << iters_man << std::endl;
      std::cout << "ERROR wrong answer dev sync" << std::endl;
    }
    dev_time = time.time();
  }/**/
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

  for (int i = 511; i < 2048; i += 64) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    int vars = 512;
    int ineqs = i;
    while (test_num < 1) {

      if (testingSimplex(vars, ineqs, test_num, cpu_time, dev_time)) {
        continue;
      }
      test_num++;
    }
    results_cpu << cpu_time << std::endl;
    results_dev << dev_time << std::endl;
    std::cout << vars + 1 << "x" << ineqs + 1 << ' ' << cpu_time << ' ' << dev_time << std::endl;
  }

  results_cpu << std::endl;
  results_dev << std::endl;
  for (int i = 512; i < 2049; i += 64) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    int vars = i;
    int ineqs = 2048;
    while (test_num < 1) {

      if (testingSimplex(vars, ineqs, test_num, cpu_time, dev_time)) {
        continue;
      }
      test_num++;
    }
    results_cpu << cpu_time << std::endl;
    results_dev << dev_time << std::endl;
    std::cout << vars + 1 << "x" << ineqs + 1 << ' ' << cpu_time << ' ' << dev_time << std::endl;
  }/**/

  results_cpu << std::endl;
  results_dev << std::endl;
  for (int i = 511; i < 1024; i += 64) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    int vars = i + 1;
    int ineqs = i;
    while (test_num < 1) {

      if (testingSimplex(vars, ineqs, test_num, cpu_time, dev_time)) {
        continue;
      }
      test_num++;
    }
    results_cpu << cpu_time << std::endl;
    results_dev << dev_time << std::endl;
    std::cout << vars + 1 << "x" << ineqs + 1 << ' ' << cpu_time << ' ' << dev_time << std::endl;
  }

  return 0;

}
