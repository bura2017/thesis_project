#include "BranchAndCut.h"
#include "MatrixMultip.h"
#include <iomanip>

void initMatrix(Matrix &matrix, const Matrix &input, taskTree * &task) {
  //std::cout << "Init new matrix..." << std::endl;
  CHECK_NULL(task);
  matrix = input;

  int *branches = new int[matrix.cols];
  for (int i = 0; i < matrix.cols; i++) {
    branches[i] = 0;
  }
  int cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    branches[branch->point] = branch->num * 2 - 1;
    cut_rows++;
    if (branch->cuts != NULL) {
      cut_rows++;
    }
  }
  //for (int i = 0; i < matrix.cols; i++) {
  //  std::cout << std::setw(2) << branches[i] << ' ';
  //}
  //Zstd::cout << std::endl;
  delete [] branches;

  if (!cut_rows) {
    CHECK_NULL(NULL);
  }
  Matrix cuts (cut_rows, input.cols);
  cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    int point = branch->point;
    double value = branch->value;
    if (branch->num == 0) {
      cuts.e[cut_rows + point * cuts.m] = 1.0;
      cuts.e[cut_rows + 0 * cuts.m] = value;
    } else {
      cuts.e[cut_rows + point * cuts.m] = -1.0;
      cuts.e[cut_rows + 0 * cuts.m] = - value;
    }
    cut_rows++;
    if (branch->cuts != NULL) {
      for (int j = 0; j < matrix.cols; j++) {
        cuts.e[cut_rows + j * cuts.m] = branch->cuts[j];
      }
      cut_rows ++;
    }
  }
  //Matrix temp_matrix(cuts.rows, cuts.cols);
  //Matrix check_matrix(cuts.rows, cuts.cols);
  //MatMul(cuts, dev_trans, temp_matrix);                   //35356.4
  //cublas_multip(cuts, dev_trans, temp_matrix); //36686.2

  matrix.add_cuts(cuts);
}
