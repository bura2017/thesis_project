#include "Matrix.h"
#include "HandleError.h"
#include "BranchAndCut.h"
#include "CudaDeviceProperties.h"
#include "Time.h"
#include "GenTest.h"

#include <cstdlib>
#include <iostream>
#include <cstring>

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

  int test_num = 0;
  flag = time(NULL);
  //std::cout << flag << std::endl;

  while (test_num < 10) {
    const int vars = 200;
    const int ineqs = 300;

    //gen_test(test_num, vars, ineqs, flag);

    char filename[MAX_LENG];
    //strcpy (filename, test_files[test_num]);
    sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);
    //std::cout << filename << std::endl;
    //std::cin >> filename;
    if (filename[0] == '0') {
      return 0;
    }

    char fullname[MAX_LENG];
    sprintf(fullname, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/Type1/%s", filename);
    //sprintf(fullname, "TestGenerator/%s", filename);

    Matrix input(fullname);

    //Simplex testing
    {
      double time = 0;

      int iters_man = 0;
      {
        time = mtime();
        Matrix matrix(input);
        int iters_cpu = cpuDualSimplex (matrix);
        time = mtime() - time;
        iters_man = iters_cpu;
        std::cout << iters_man << std::endl;
        std::cout << time << std::endl;
        if (iters_cpu != iters_man) {
          std::cout << iters_cpu << " != " << iters_man << std::endl;
          std::cout << "ERROR wrong answer cpu" << std::endl;
        }
      }/**/

      /*{
        time = mtime();
        Matrix matrix(input);
        int iters_async = gpuDualSimplexAsync (matrix);
        time = mtime() - time;
        std::cout << time << std::endl;
        //std::cout << " GPU async time " << time << std::endl;
        if (iters_async != iters_man) {
          std::cout << iters_async << " != " << iters_man << std::endl;
          std::cout << "ERROR wrong answer async" << std::endl;
        }
      }/**/

      /*{
        time = mtime();
        Matrix matrix(input);
        d_matrix trans_matrix;
        trans_matrix.rows = matrix.cols;
        trans_matrix.cols = matrix.cols;
        trans_matrix.m = matrix.cols;
        CHECK_CUDA(cudaMalloc (&trans_matrix.e, sizeof(double) * trans_matrix.m * trans_matrix.cols));
        int iters_async = gpuDualSimplexAsync (matrix, trans_matrix);
        cudaFree(trans_matrix.e);
        time = mtime() - time;
        std::cout << time << std::endl;
        //std::cout << " GPU async time " << time << std::endl;
        if (iters_async != iters_man) {
          std::cout << iters_async << " != " << iters_man << std::endl;
          std::cout << "ERROR wrong answer async" << std::endl;
        }
      }/**/

      /*{
        time = mtime();
        Matrix matrix(input);
        int iters_sync = gpuDualSimplexSync (matrix);
        time = mtime() - time;
        std::cout << time << std::endl;
        if (iters_sync != iters_man) {
          std::cout << iters_sync << " != " << iters_man << std::endl;
          std::cout << "ERROR wrong answer sync" << std::endl;
        }
      }/**/

      /*{
        time = mtime();
        Matrix matrix(input);
        int iters_sync_dev = gpuDualSimplexSyncDev (matrix);
        time = mtime() - time;
        std::cout << time << std::endl;
        if (iters_sync_dev != iters_man) {
          std::cout << iters_sync_dev << " != " << iters_man << std::endl;
          std::cout << "ERROR wrong answer sync dev" << std::endl;
        }
      }/**/

    }

    //BranchAndCut testing
    /*{
      double time = mtime();
      std::cout << (branchAndCut(input) ? "sat" : "unsat");
      time = mtime() - time;
      std::cout << " time " << time << std::endl << std::endl;
    }/**/

    test_num++;
    return 0;
  }

}
