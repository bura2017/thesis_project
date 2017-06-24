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

    double cpu_time = 0.0;
    double gpu_time = 0.0;
    int test_num = 0;
    flag = time(NULL);
    while (test_num < 1) {
      int vars = 200;
      int ineqs = 300;

      gen_test(test_num, vars, ineqs, flag);

      char filename[MAX_LENG];
      sprintf(filename, "Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

      char fullname[MAX_LENG];
      sprintf(fullname, "TestGenerator/%s", filename);

      Matrix input(fullname);

      cuda_time time;
      cmp ((double) input.cols, (double) input.rows);

      //====================BRANCH AND BOUND========================

      {
        Matrix matrix(input);
        time.start();
        std::cout << (branchAndBound(matrix) > 0 ? "sat" : "unsat");
        time.stop();
        std::cout << " time " << time.time() << std::endl << std::endl;
      }/**/
      test_num++;
    }
}
