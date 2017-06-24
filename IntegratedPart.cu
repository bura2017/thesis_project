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

int flag;

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

  std::ofstream results("Results.txt");

  for (int i = 100; i < 501; i += 10) {
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 5) {
      int vars = i;
      int ineqs = i;

      gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);

      Matrix input(fullname);

      cuda_time time;
      cmp ((double) input.cols, (double) input.rows);

      //====================SIMPLEX TESTING=========================
      {
        int iters_man = 0;

        {
          time.start();
          Matrix matrix(input);
          int iters_cpu = cpuDualSimplex (matrix);
          time.stop();
          iters_man = iters_cpu;
          if (iters_man == 0) {
            continue;
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
          cpu_time += time.time() / iters_cpu;
        }/**/

        {
          time.start();
          Matrix matrix(input);
          int iters_async = gpuDualSimplexAsync (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_async != iters_man) {
            std::cout << iters_async << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer async" << std::endl;
          }
          gpu_time += time.time() / iters_async;
        }/**/

      }

      //====================BRANCH AND BOUND========================

      /*{
        Matrix matrix(input);
        time.start();
        std::cout << (branchAndBound(matrix) ? "sat" : "unsat");
        time.stop();
        std::cout << " time " << time.time() << std::endl << std::endl;
      }/**/
      test_num++;
    }
    results << cpu_time / gpu_time << std::endl;
  }
}
