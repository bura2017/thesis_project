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

  std::ofstream results("Results.txt");

  for (int i = 100; i < 1501; i += 50) {
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = 400;
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
          std::cout << iters_man << std::endl;
          std::cout << time.time() << std::endl;
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
          int pow = 2;
          for (int i = 1; i > 0; i++) {
            if (input.rows > pow && input.rows < pow * 2) {
              break;
            }
            pow *= 2;
          }
          pow *= 2;
          time.start();
          Matrix matrix(input, pow - input.rows);
          int iters_async = gpuDualSimplexAsync (matrix);
          time.stop();
          std::cout << time.time() << std::endl;
          if (iters_async != iters_man) {
            std::cout << iters_async << " != " << iters_man << std::endl;
            std::cout << "ERROR wrong answer async" << std::endl;
          }
          if (iters_async > 0) {
            gpu_time += time.time() / iters_async;
          } else {
            gpu_time -= time.time() / iters_async;
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
      //return 0;
    }
    results << cpu_time / gpu_time << std::endl;
    std::cout << "Result " << cpu_time / gpu_time << std::endl;
  }

  //return 0;

  results << std::endl;
  for (int i = 100; i < 701; i += 50) {
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = i * 2;

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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

        }/**/

        {
          int pow = 2;
          for (int i = 1; i > 0; i++) {
            if (input.rows > pow && input.rows < pow * 2) {
              break;
            }
            pow *= 2;
          }
          pow *= 2;
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
            gpu_time += time.time() / iters_async;
          } else {
            gpu_time -= time.time() / iters_async;
          }
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
    std::cout << "Result " << cpu_time / gpu_time << std::endl;
  }

  results << std::endl;
  for (int i = 100; i < 801; i += 50) {
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = i;
      int ineqs = 1000 - i;

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
          if (iters_cpu > 0) {
            cpu_time += time.time() / iters_cpu;
          } else {
            cpu_time -= time.time() / iters_cpu;
          }

        }/**/

        {
          int pow = 2;
          for (int i = 1; i > 0; i++) {
            if (input.rows > pow && input.rows < pow * 2) {
              break;
            }
            pow *= 2;
          }
          pow *= 2;
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
            gpu_time += time.time() / iters_async;
          } else {
            gpu_time -= time.time() / iters_async;
          }
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
    std::cout << "Result " << cpu_time / gpu_time << std::endl;
  }

}
