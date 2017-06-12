#include <fstream>

#define MAX_LEN 1024

void gen_test(int test_num, int vars, int ineqs, int &flag) {
  flag ++;
  srand (flag);
  char filename[MAX_LEN];
  const int cols = vars + 1;
  const int rows = 1 + vars + ineqs;
  sprintf (filename, "/home/valerius/cuda-workspace/Benchmarks_txt/TestGenerator/Type1/Vars-%d_Ineqs-%d_%d.ilp", vars, ineqs, test_num);

  std::ofstream of (filename);
  if (!of.is_open()) {
    ERROR("Can't open file");
  }
  of << cols << ' ' << rows << std::endl;

  for (int i = 0; i < cols; i++) {
    of << "1 1" << std::endl;
  }
  for (int i = 1; i < cols; i++) {
    for (int j = 0; j < cols; j++) {
      if (i == j) {
        of << "-1 1" << std::endl;
      } else {
        of << "0 1" << std::endl;
      }
    }
  }/**/
  int *solution = new int[vars];
  for (int i = 0; i < vars; i++) {
    solution[i] = rand() % 10;
  }
  int *vec = new int[vars];

  for (int i = 0; i < ineqs; i++) {
    int first = 1;
    for (int j = 0; j < vars; j ++) {
      vec[j] = rand() % 10 - 1;
      if (vec[j] > 1) {
        vec[j] = 0;
      }
      first += vec[j] * solution[j];
    }
    of << first << " 1" << std::endl;
    for (int j = 0; j < vars; j++) {
      of << vec[j] << " 1" << std::endl;
    }
  }
  delete [] solution;
  delete [] vec;

}
