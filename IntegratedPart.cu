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

  std::ofstream results_dev("results_dev.txt");

  for (int i = 500; i < 1501; i += 50) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = 500;
      int ineqs = i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      //sprintf(fullname, "TestGenerator/%s", filename);
      if (access(fullname, 0) == -1) {
        gen_test(test_num, vars, ineqs, flag);
      }

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
            //continue;
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

        }/**/

        {
          int supply = input.rows % BLOCK_SIZE;
          supply = (supply > 0 ? BLOCK_SIZE - supply : 0);
          time.start();
          Matrix matrix(input, supply);
          int iters_dev = gpuDualSimplexSyncDev (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_dev != iters_man) {
            std::cout << iters_dev << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer dev sync" << std::endl;
          }
          if (iters_dev > 0) {
            dev_time += time.time() / iters_dev;
          } else {
            dev_time -= time.time() / iters_dev;
          }
        }/**/

      }
      test_num++;
    }
    //results_sync << cpu_time / sync_time << std::endl;
    //results_async << cpu_time / async_time << std::endl;
    results_dev << cpu_time / dev_time << std::endl;
    std::cout << "Result dev sync " << cpu_time / dev_time << std::endl;
  }

  results_dev << std::endl;
  for (int i = 500; i < 1501; i += 50) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = 2000 - i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      //sprintf(fullname, "TestGenerator/%s", filename);
      if (access(fullname, 0) == -1) {
        gen_test(test_num, vars, ineqs, flag);
      }

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
            //continue;
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

        }/**/

        {
          int supply = input.rows % BLOCK_SIZE;
          supply = (supply > 0 ? BLOCK_SIZE - supply : 0);
          time.start();
          Matrix matrix(input, supply);
          int iters_dev = gpuDualSimplexSyncDev (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_dev != iters_man) {
            std::cout << iters_dev << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer dev sync" << std::endl;
          }
          if (iters_dev > 0) {
            dev_time += time.time() / iters_dev;
          } else {
            dev_time -= time.time() / iters_dev;
          }
        }/**/

      }

      test_num++;
    }
    results_dev << cpu_time / dev_time << std::endl;
    std::cout << "Result dev sync " << cpu_time / dev_time << std::endl;
  }

  results_dev << std::endl;
  for (int i = 500; i < 1001; i += 50) {
    double cpu_time = 0.0;
    double dev_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      //sprintf(fullname, "TestGenerator/%s", filename);
      if (access(fullname, 0) == -1) {
        gen_test(test_num, vars, ineqs, flag);
      }

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
            //continue;
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

        }/**/

        {
          int supply = input.rows % BLOCK_SIZE;
          supply = (supply > 0 ? BLOCK_SIZE - supply : 0);
          time.start();
          Matrix matrix(input, supply);
          int iters_dev = gpuDualSimplexSyncDev (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_dev != iters_man) {
            std::cout << iters_dev << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer dev sync" << std::endl;
          }
          if (iters_dev > 0) {
            dev_time += time.time() / iters_dev;
          } else {
            dev_time -= time.time() / iters_dev;
          }
        }/**/

      }

      test_num++;
    }
    results_dev << cpu_time / dev_time << std::endl;
    std::cout << "Result dev sync " << cpu_time / dev_time << std::endl;
  }


  return 0;

}
