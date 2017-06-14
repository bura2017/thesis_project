#include "BranchAndCut.h"
#include "MatrixMultip.h"

void initMatrix(Matrix &matrix, const Matrix &input, taskTree * &task, d_matrix &dev_trans) {
  //std::cout << "Init new matrix..." << std::endl;
  CHECK_NULL(task);
  matrix = input;

  int cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    cut_rows++;
  }
  if (!cut_rows) {
    CHECK_NULL(NULL);
  }
  Matrix cuts (cut_rows, input.cols);
  cut_rows = 0;
  for (taskTree *branch = task; branch->prev != NULL; branch = branch->prev) {
    int point = branch->prev->point;
    double value = branch->prev->value;
    if (branch->num == 0) {
      cuts.e[cut_rows + point * cuts.m] = 1.0;
      cuts.e[cut_rows + 0 * cuts.m] = value;
    } else {
      cuts.e[cut_rows + point * cuts.m] = -1.0;
      cuts.e[cut_rows + 0 * cuts.m] = -(value + 1.0);
    }
    cut_rows++;
  }
  Matrix temp_matrix(cuts.rows, cuts.cols);
  //Matrix check_matrix(cuts.rows, cuts.cols);
  //MatMul(cuts, dev_trans, temp_matrix);                   //35356.4
  cublas_multip(cuts, dev_trans, temp_matrix); //36686.2

  matrix.add_cuts(temp_matrix);
}
