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

  std::ofstream results_sync("results_sync.txt");
  std::ofstream results_async("results_async.txt");

  for (int i = 100; i < 1001; i += 50) {
    double cpu_time = 0.0;
    double sync_time = 0.0;
    double async_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = 200;
      int ineqs = i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      //sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      sprintf(fullname, "TestGenerator/%s", filename);
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

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
          if (iters_async > 0) {
            async_time += time.time() / iters_async;
          } else {
            async_time -= time.time() / iters_async;
          }
        }/**/

        {
          time.start();
          Matrix matrix(input);
          int iters_sync = gpuDualSimplexSync (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_sync != iters_man) {
            std::cout << iters_sync << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer sync" << std::endl;
          }
          if (iters_sync > 0) {
            sync_time += time.time() / iters_sync;
          } else {
            sync_time -= time.time() / iters_sync;
          }
        }/**/

      }

      //====================BRANCH AND BOUND========================

      /*{
        Matrix matrix(input);
        time.start();
        std::cout << (branchAndBound(matrix) > 0 ? "sat" : "unsat");
        time.stop();
        std::cout << " time " << time.time() << std::endl << std::endl;
      }/**/
      test_num++;
    }
    results_sync << cpu_time / sync_time << std::endl;
    results_sync << cpu_time / async_time << std::endl;
    std::cout << "Result sync " << cpu_time / sync_time  << " async " << cpu_time / async_time << std::endl;
  }

  for (int i = 100; i < 501; i += 50) {
    double cpu_time = 0.0;
    double sync_time = 0.0;
    double async_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      //sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      sprintf(fullname, "TestGenerator/%s", filename);
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

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
          if (iters_async > 0) {
            async_time += time.time() / iters_async;
          } else {
            async_time -= time.time() / iters_async;
          }
        }/**/

        {
          time.start();
          Matrix matrix(input);
          int iters_sync = gpuDualSimplexSync (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_sync != iters_man) {
            std::cout << iters_sync << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer sync" << std::endl;
          }
          if (iters_sync > 0) {
            sync_time += time.time() / iters_sync;
          } else {
            sync_time -= time.time() / iters_sync;
          }
        }/**/

      }

      //====================BRANCH AND BOUND========================

      /*{
        Matrix matrix(input);
        time.start();
        std::cout << (branchAndBound(matrix) > 0 ? "sat" : "unsat");
        time.stop();
        std::cout << " time " << time.time() << std::endl << std::endl;
      }/**/
      test_num++;
    }
    results_sync << cpu_time / sync_time << std::endl;
    results_sync << cpu_time / async_time << std::endl;
    std::cout << "Result sync " << cpu_time / sync_time  << " async " << cpu_time / async_time << std::endl;
  }

  for (int i = 100; i < 501; i += 50) {
    double cpu_time = 0.0;
    double sync_time = 0.0;
    double async_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = 1000 - i;

      //gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      //sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/%s", filename);
      sprintf(fullname, "TestGenerator/%s", filename);
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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

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
          if (iters_async > 0) {
            async_time += time.time() / iters_async;
          } else {
            async_time -= time.time() / iters_async;
          }
        }/**/

        {
          time.start();
          Matrix matrix(input);
          int iters_sync = gpuDualSimplexSync (matrix);
          time.stop();
          //std::cout << time.time() << std::endl;
          if (iters_sync != iters_man) {
            std::cout << iters_sync << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer sync" << std::endl;
          }
          if (iters_sync > 0) {
            sync_time += time.time() / iters_sync;
          } else {
            sync_time -= time.time() / iters_sync;
          }
        }/**/

      }

      //====================BRANCH AND BOUND========================

      /*{
        Matrix matrix(input);
        time.start();
        std::cout << (branchAndBound(matrix) > 0 ? "sat" : "unsat");
        time.stop();
        std::cout << " time " << time.time() << std::endl << std::endl;
      }/**/
      test_num++;
    }
    results_sync << cpu_time / sync_time << std::endl;
    results_sync << cpu_time / async_time << std::endl;
    std::cout << "Result sync " << cpu_time / sync_time  << " async " << cpu_time / async_time << std::endl;
  }

  return 0;

}
