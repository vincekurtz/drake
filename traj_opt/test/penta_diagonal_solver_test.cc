#include "drake/traj_opt/penta_diagonal_solver.h"

#include <chrono>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/traj_opt/penta_diagonal_matrix.h"
#include "drake/traj_opt/penta_diagonal_to_petsc_matrix.h"
#include "drake/multibody/fem/petsc_symmetric_block_sparse_matrix.h"

using drake::multibody::fem::internal::PetscSymmetricBlockSparseMatrix;
using drake::multibody::fem::internal::PetscSolverStatus;
using std::chrono::steady_clock;

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace drake {
namespace traj_opt {
namespace internal {

GTEST_TEST(PentaDiagonalMatrixTest, MultiplyBy) {
  // Generate a random penta-diagonal matrix
  const int block_size = 2;
  const int num_blocks = 5;
  const int size = num_blocks * block_size;
  const MatrixXd A = MatrixXd::Random(size, size);
  const PentaDiagonalMatrix<double> H =
      PentaDiagonalMatrix<double>::MakeSymmetricFromLowerDense(A, num_blocks,
                                                               block_size);

  // Multiply by an arbitrary vector
  const VectorXd v = VectorXd::LinSpaced(size, 0.1, 1.1);
  VectorXd prod(size);

  H.MultiplyBy(v, &prod);
  const VectorXd prod_expected = H.MakeDense() * v;

  const double kTolerance = std::numeric_limits<double>::epsilon() * size;
  EXPECT_TRUE(CompareMatrices(prod, prod_expected, kTolerance,
                              MatrixCompareType::relative));
}

GTEST_TEST(PentaDiagonalMatrixTest, SymmetricMatrixEmpty) {
  const std::vector<MatrixXd> empty_diagonal;
  PentaDiagonalMatrix<double> M(empty_diagonal, empty_diagonal, empty_diagonal);
  EXPECT_EQ(M.rows(), 0);
}

GTEST_TEST(PentaDiagonalMatrixTest, MutateMatrix) {
  const int k = 3;
  PentaDiagonalMatrix<double> M(5, k);
  EXPECT_TRUE(M.is_symmetric());
  EXPECT_EQ(M.block_rows(), 5);
  EXPECT_EQ(M.block_cols(), 5);
  EXPECT_EQ(M.block_size(), 3);
  EXPECT_EQ(M.rows(), 15);
  EXPECT_EQ(M.cols(), 15);

  const MatrixXd B1 = 1.5 * MatrixXd::Ones(k, k);
  const MatrixXd B2 = 2.1 * MatrixXd::Ones(k, k);
  const MatrixXd B3 = -12.8 * MatrixXd::Ones(k, k);
  const MatrixXd B4 = 1.8 * MatrixXd::Ones(k, k);
  const MatrixXd B5 = 15.3 * MatrixXd::Ones(k, k);
  const std::vector<MatrixXd> some_diagonal = {B1, B2, B3, B3, B5};

  // These throw since M is diagonal and it only allows mutating the lower
  // diagonals.
  EXPECT_THROW(M.mutable_D(), std::exception);
  EXPECT_THROW(M.mutable_E(), std::exception);

  // Mutate diagonals.
  EXPECT_NE(M.A(), some_diagonal);
  M.mutable_A() = some_diagonal;
  EXPECT_EQ(M.A(), some_diagonal);

  EXPECT_NE(M.B(), some_diagonal);
  M.mutable_B() = some_diagonal;
  EXPECT_EQ(M.B(), some_diagonal);

  EXPECT_NE(M.C(), some_diagonal);
  M.mutable_C() = some_diagonal;
  EXPECT_EQ(M.C(), some_diagonal);

  // We've changed some terms in the matrix, so we can no longer assume it's
  // symmetric
  EXPECT_FALSE(M.is_symmetric());
}

GTEST_TEST(PentaDiagonalMatrixTest, SymmetricMatrix) {
  const int k = 5;
  const MatrixXd Z = 1.5 * MatrixXd::Zero(k, k);
  const MatrixXd B1 = 1.5 * MatrixXd::Ones(k, k);
  const MatrixXd B2 = 2.1 * MatrixXd::Ones(k, k);
  const MatrixXd B3 = -12.8 * MatrixXd::Ones(k, k);
  const MatrixXd B4 = 1.8 * MatrixXd::Ones(k, k);
  const MatrixXd B5 = 15.3 * MatrixXd::Ones(k, k);
  const MatrixXd B6 = 7.1 * MatrixXd::Ones(k, k);
  PentaDiagonalMatrix<double> M({Z, Z, B1}, {Z, B2, B3}, {B4, B5, B6});
  EXPECT_EQ(M.rows(), k * 3);
  EXPECT_EQ(M.block_rows(), 3);
  // Verify M is symmetric and is properly zero padded.
  EXPECT_EQ(M.D()[0], M.B()[1]);
  EXPECT_EQ(M.D()[1], M.B()[2]);
  EXPECT_EQ(M.D()[2], Z);
  EXPECT_EQ(M.E()[0], M.A()[2]);
  EXPECT_EQ(M.E()[1], Z);
  EXPECT_EQ(M.E()[2], Z);
}

GTEST_TEST(PentaDiagonalMatrixTest, SolveIdentity) {
  const int block_size = 3;
  const int num_blocks = 5;
  const int size = num_blocks * block_size;
  const PentaDiagonalMatrix<double> H =
      PentaDiagonalMatrix<double>::MakeIdentity(num_blocks, block_size);
  PentaDiagonalFactorization Hlu(H);
  EXPECT_EQ(Hlu.status(), PentaDiagonalFactorizationStatus::kSuccess);

  const VectorXd b = VectorXd::LinSpaced(size, -3, 12.4);
  VectorXd x = b;
  Hlu.SolveInPlace(&x);

  EXPECT_EQ(x, b);
}

GTEST_TEST(PentaDiagonalMatrixTest, SolveBlockDiagonal) {
  const int block_size = 3;
  const int num_blocks = 5;
  const int size = num_blocks * block_size;
  const MatrixXd I = MatrixXd::Identity(block_size, block_size);
  const MatrixXd Z = MatrixXd::Zero(block_size, block_size);
  const MatrixXd random_block = MatrixXd::Random(block_size, block_size);
  const MatrixXd B1 = 2.1 * I + random_block * random_block.transpose();
  const MatrixXd B2 = 3.5 * I + random_block * random_block.transpose();
  const MatrixXd B3 = 0.2 * I + random_block * random_block.transpose();

  std::vector<MatrixXd> A(num_blocks, Z);
  std::vector<MatrixXd> B(num_blocks, Z);
  std::vector<MatrixXd> C{B1, B2, B3, B1, B3};
  const PentaDiagonalMatrix<double> H(std::move(A), std::move(B), std::move(C));
  const MatrixXd Hdense = H.MakeDense();

  PentaDiagonalFactorization Hlu(H);
  EXPECT_EQ(Hlu.status(), PentaDiagonalFactorizationStatus::kSuccess);
  const VectorXd b = VectorXd::LinSpaced(size, -3, 12.4);
  VectorXd x = b;
  Hlu.SolveInPlace(&x);

  // Reference solution computed with Eigen, dense.
  const VectorXd x_expected = Hdense.ldlt().solve(b);

  const double kTolerance = std::numeric_limits<double>::epsilon() * size;
  EXPECT_TRUE(
      CompareMatrices(x, x_expected, kTolerance, MatrixCompareType::relative));
}

GTEST_TEST(PentaDiagonalMatrixTest, SolveTriDiagonal) {
  const int block_size = 3;
  const int num_blocks = 5;
  const int size = num_blocks * block_size;
  const MatrixXd I = MatrixXd::Identity(block_size, block_size);
  const MatrixXd Z = MatrixXd::Zero(block_size, block_size);
  const MatrixXd random_block = MatrixXd::Random(block_size, block_size);
  const MatrixXd B1 = 2.1 * I + random_block * random_block.transpose();
  const MatrixXd B2 = 3.5 * I + random_block * random_block.transpose();
  const MatrixXd B3 = 0.2 * I + random_block * random_block.transpose();
  const MatrixXd B4 = 1.3 * I + random_block * random_block.transpose();

  std::vector<MatrixXd> A(num_blocks, Z);
  std::vector<MatrixXd> B{Z, B1, B2, B3, B4};
  std::vector<MatrixXd> C{B1, B2, B3, B1, B3};
  const PentaDiagonalMatrix<double> H(std::move(A), std::move(B), std::move(C));
  const MatrixXd Hdense = H.MakeDense();

  PentaDiagonalFactorization Hlu(H);
  EXPECT_EQ(Hlu.status(), PentaDiagonalFactorizationStatus::kSuccess);
  const VectorXd b = VectorXd::LinSpaced(size, -3, 12.4);
  VectorXd x = b;
  Hlu.SolveInPlace(&x);

  // Reference solution computed with Eigen, dense.
  const VectorXd x_expected = Hdense.ldlt().solve(b);

  const double kTolerance = std::numeric_limits<double>::epsilon() * size;
  EXPECT_TRUE(
      CompareMatrices(x, x_expected, kTolerance, MatrixCompareType::relative));
}

GTEST_TEST(PentaDiagonalMatrixTest, SolvePentaDiagonal) {
  const int block_size = 2;
  const int num_blocks = 21;
  const int size = num_blocks * block_size;

  bool random_H = false;  // we'll solve H*x = b

  // Generate an SPD matrix.
  MatrixXd P(size, size);
  if (random_H) {
    const MatrixXd A = 1e4 * MatrixXd::Random(size, size);
    P = MatrixXd::Identity(size, size) + A * A.transpose();
  } else {
    P << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.21717e+09, 6.07423e+08,
        -8.1363e+08, -4.05443e+08, 2.0499e+08, 1.01962e+08, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 6.07423e+08, 3.03132e+08, -4.06423e+08, -2.02526e+08,
        1.02429e+08, 5.09482e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -8.1363e+08, -4.06423e+08, 1.21689e+09, 6.07295e+08, -8.13172e+08,
        -4.0526e+08, 2.04678e+08, 1.01837e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -4.05443e+08, -2.02526e+08, 6.07295e+08, 3.03075e+08, -4.06194e+08,
        -2.02435e+08, 1.02273e+08, 5.08857e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.0499e+08, 1.02429e+08, -8.13172e+08, -4.06194e+08, 1.21642e+09,
        6.07086e+08, -8.12546e+08, -4.0501e+08, 2.04289e+08, 1.01681e+08, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1.01962e+08, 5.09482e+07, -4.0526e+08, -2.02435e+08,
        6.07086e+08, 3.02982e+08, -4.05882e+08, -2.0231e+08, 1.02079e+08,
        5.08081e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.04678e+08, 1.02273e+08,
        -8.12546e+08, -4.05882e+08, 1.21578e+09, 6.068e+08, -8.11769e+08,
        -4.04699e+08, 2.03834e+08, 1.01499e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.01837e+08, 5.08857e+07, -4.0501e+08, -2.0231e+08, 6.068e+08,
        3.02856e+08, -4.05493e+08, -2.02155e+08, 1.01852e+08, 5.07172e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2.04289e+08, 1.02079e+08, -8.11769e+08,
        -4.05493e+08, 1.215e+09, 6.06446e+08, -8.10858e+08, -4.04335e+08,
        2.03324e+08, 1.01295e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.01681e+08,
        5.08081e+07, -4.04699e+08, -2.02155e+08, 6.06446e+08, 3.02699e+08,
        -4.05039e+08, -2.01973e+08, 1.01597e+08, 5.06152e+07, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2.03834e+08, 1.01852e+08, -8.10858e+08, -4.05039e+08,
        1.21407e+09, 6.06032e+08, -8.09838e+08, -4.03926e+08, 2.02772e+08,
        1.01074e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.01499e+08, 5.07172e+07,
        -4.04335e+08, -2.01973e+08, 6.06032e+08, 3.02515e+08, -4.04529e+08,
        -2.01769e+08, 1.01321e+08, 5.05048e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.03324e+08, 1.01597e+08, -8.09838e+08, -4.04529e+08, 1.21304e+09,
        6.05569e+08, -8.08732e+08, -4.03484e+08, 2.0219e+08, 1.00841e+08, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1.01295e+08, 5.06152e+07, -4.03926e+08, -2.01769e+08,
        6.05569e+08, 3.0231e+08, -4.03977e+08, -2.01548e+08, 1.0103e+08,
        5.03885e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.02772e+08, 1.01321e+08,
        -8.08732e+08, -4.03977e+08, 1.21193e+09, 6.05068e+08, -8.07568e+08,
        -4.03018e+08, 2.01594e+08, 1.00603e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.01074e+08, 5.05048e+07, -4.03484e+08, -2.01548e+08, 6.05068e+08,
        3.02088e+08, -4.03396e+08, -2.01315e+08, 1.00732e+08, 5.02693e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2.0219e+08, 1.0103e+08, -8.07568e+08,
        -4.03396e+08, 1.21076e+09, 6.04543e+08, -8.06374e+08, -4.02541e+08,
        2.00997e+08, 1.00364e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.00841e+08,
        5.03885e+07, -4.03018e+08, -2.01315e+08, 6.04543e+08, 3.01854e+08,
        -4.028e+08, -2.01077e+08, 1.00434e+08, 5.01501e+07, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 2.01594e+08, 1.00732e+08, -8.06374e+08, -4.028e+08,
        1.20956e+09, 6.04005e+08, -8.05181e+08, -4.02063e+08, 2.00416e+08,
        1.00132e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.00603e+08, 5.02693e+07,
        -4.02541e+08, -2.01077e+08, 6.04005e+08, 3.01616e+08, -4.02204e+08,
        -2.00839e+08, 1.00144e+08, 5.00339e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.00997e+08, 1.00434e+08, -8.05181e+08, -4.02204e+08, 1.20837e+09,
        6.03469e+08, -8.04017e+08, -4.01598e+08, 1.99863e+08, 9.99106e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1.00364e+08, 5.01501e+07, -4.02063e+08, -2.00839e+08,
        6.03469e+08, 3.01377e+08, -4.01622e+08, -2.00606e+08, 9.98676e+07,
        4.99234e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.00416e+08, 1.00144e+08,
        -8.04017e+08, -4.01622e+08, 1.20721e+09, 6.02948e+08, -8.02911e+08,
        -4.01155e+08, 1.99353e+08, 9.97066e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.00132e+08, 5.00339e+07, -4.01598e+08, -2.00606e+08, 6.02948e+08,
        3.01146e+08, -4.0107e+08, -2.00385e+08, 9.96127e+07, 4.98214e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1.99863e+08, 9.98676e+07, -8.02911e+08,
        -4.0107e+08, 1.20611e+09, 6.02453e+08, -8.0189e+08, -4.00747e+08,
        1.98898e+08, 9.95247e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.99106e+07,
        4.99234e+07, -4.01155e+08, -2.00385e+08, 6.02453e+08, 3.00926e+08,
        -4.0056e+08, -2.00181e+08, 9.93854e+07, 4.97305e+07, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1.99353e+08, 9.96127e+07, -8.0189e+08, -4.0056e+08,
        1.2051e+09, 6.01998e+08, -8.0098e+08, -4.00383e+08, 1.9851e+08,
        9.93693e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.97066e+07, 4.98214e+07,
        -4.00747e+08, -2.00181e+08, 6.01998e+08, 3.00724e+08, -4.00106e+08,
        -2e+08, 9.91913e+07, 4.96529e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.98898e+08, 9.93854e+07, -8.0098e+08, -4.00106e+08, 1.2042e+09,
        6.01593e+08, -8.00202e+08, -4.00072e+08, 1.98197e+08, 9.92442e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 9.95247e+07, 4.97305e+07, -4.00383e+08, -2e+08,
        6.01593e+08, 3.00544e+08, -3.99718e+08, -1.99844e+08, 9.90351e+07,
        4.95904e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.9851e+08, 9.91913e+07,
        -8.00202e+08, -3.99718e+08, 1.20343e+09, 6.01248e+08, -7.99577e+08,
        -3.99822e+08, 1.97968e+08, 9.91526e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        9.93693e+07, 4.96529e+07, -4.00072e+08, -1.99844e+08, 6.01248e+08,
        3.0039e+08, -3.99405e+08, -1.99719e+08, 9.89207e+07, 4.95446e+07, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1.98197e+08, 9.90351e+07, -7.99577e+08,
        -3.99405e+08, 1.20282e+09, 6.0097e+08, -7.99119e+08, -3.99638e+08,
        1.97828e+08, 9.90968e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.92442e+07,
        4.95904e+07, -3.99822e+08, -1.99719e+08, 6.0097e+08, 3.00267e+08,
        -3.99176e+08, -1.99628e+08, 9.88509e+07, 4.95167e+07, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1.97968e+08, 9.89207e+07, -7.99119e+08, -3.99176e+08,
        1.20237e+09, 6.00768e+08, -7.98839e+08, -3.99527e+08, 1.97781e+08,
        9.9078e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.91526e+07, 4.95446e+07,
        -3.99638e+08, -1.99628e+08, 6.00768e+08, 3.00177e+08, -3.99037e+08,
        -1.99572e+08, 9.88274e+07, 4.95073e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.97828e+08, 9.88509e+07, -7.98839e+08, -3.99037e+08, 1.00076e+09,
        5.00041e+08, -3.95818e+08, -1.98283e+08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        9.90968e+07, 4.95167e+07, -3.99527e+08, -1.99572e+08, 5.00041e+08,
        2.49853e+08, -1.97655e+08, -9.90152e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1.97781e+08, 9.88274e+07, -3.95818e+08, -1.97655e+08, 1.94292e+08,
        9.73295e+07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.9078e+07, 4.95073e+07,
        -1.98283e+08, -9.90152e+07, 9.73295e+07, 4.87577e+07;
  }

  // Generate a penta-diagonal SPD matrix. Ignore off-diagonal elements of
  // P outside the 5-diagonal band.
  const PentaDiagonalMatrix<double> H =
      PentaDiagonalMatrix<double>::MakeSymmetricFromLowerDense(P, num_blocks,
                                                               block_size);
  const MatrixXd Hdense = H.MakeDense();
  std::cout << "P error: " << (P - Hdense).norm() / P.norm() << std::endl;

  std::cout << "condition number: " << 1 / Hdense.llt().rcond() << std::endl;

  // Compute a ground truth value
  const VectorXd x_gt = VectorXd::LinSpaced(size, -3, 12.4);
  const VectorXd b = Hdense * x_gt;

  // Sanity check multiplication on the penta diagonal Hessian.
  VectorXd b2(b.size());
  H.MultiplyBy(x_gt, &b2);
  std::cout << "b error: " << (b - b2).norm() / b.norm() << std::endl;

  // Solution with dense algebra
  const auto ldlt = Hdense.ldlt();
  const VectorXd x_ldlt = ldlt.solve(b);
  std::cout << "D(ldlt): " << ldlt.vectorD().transpose() << std::endl;
  std::cout << "P(ldlt):\n"
            << ldlt.transpositionsP().indices().transpose() << std::endl;

  const VectorXd x_llt = Hdense.llt().solve(b);
  const VectorXd x_lu = Hdense.partialPivLu().solve(b);

  std::cout << "LU(partial piv.) error: " << (x_lu - x_gt).norm() / x_gt.norm()
            << std::endl;
  std::cout << "LLT error: " << (x_llt - x_gt).norm() / x_gt.norm()
            << std::endl;
  std::cout << "LDLT error: " << (x_ldlt - x_gt).norm() / x_gt.norm()
            << std::endl;

  // Solve the permuted system instead to see if stability affects round-off
  // errors.
  const MatrixXd perm = Hdense.partialPivLu().permutationP() *
                        MatrixXd::Identity(Hdense.rows(), Hdense.cols());
  const MatrixXd H_tilde = perm * Hdense * perm.transpose();
  const VectorXd b_tilde = perm * b;
  const VectorXd x_tilde = H_tilde.partialPivLu().solve(b_tilde);
  // N.B. The experiments below with a permuted sparse matrix do not work
  // because we must ensure the permutation is applied blockwise, something that
  // we cannot guarantee with the permutation obtained with Eigen's LU.
#if 0   
  const PentaDiagonalMatrix<double> H_tilde_sparse =
      PentaDiagonalMatrix<double>::MakeSymmetricFromLowerDense(
          H_tilde, num_blocks, block_size);
  PentaDiagonalFactorization Hlu_tilde(H_tilde_sparse);
  EXPECT_EQ(Hlu_tilde.status(), PentaDiagonalFactorizationStatus::kSuccess);
  VectorXd x_tilde = b_tilde;
  Hlu_tilde.SolveInPlace(&x_tilde);
#endif
  const VectorXd x2 = perm.transpose() * x_tilde;
  std::cout << "(permuted) error: " << (x2 - x_gt).norm() / x_gt.norm()
            << std::endl;

  VectorXd x_sparse = b;
  // Solution with ours pentadiagonal solver.
  steady_clock::time_point start = steady_clock::now();
  PentaDiagonalFactorization Hlu(H);
  EXPECT_EQ(Hlu.status(), PentaDiagonalFactorizationStatus::kSuccess);
  Hlu.SolveInPlace(&x_sparse);
  steady_clock::time_point end = steady_clock::now();
  double wall_clock_time = std::chrono::duration<double>(end - start).count();
  fmt::print(
      "PentaDiagonalFactorization. Wall clock: {:.4g} seconds. error: {}\n",
      wall_clock_time, (x_sparse - x_gt).norm() / x_gt.norm());

  // Solution with PetsC solver.
  start = steady_clock::now();
  auto Hpetsc = PentaDiagonalToPetscMatrix(H);
  end = steady_clock::now();
  wall_clock_time = std::chrono::duration<double>(end - start).count();
  fmt::print("PentaDiagonalToPetscMatrix(). Wall clock: {:.4g} seconds.\n",
             wall_clock_time);
  Hpetsc->set_relative_tolerance(1.0e-16);

  // Sanity check matrices are equivalent.
  const MatrixXd dense_from_pentadiagonal = H.MakeDense();
  const MatrixXd dense_from_petsc = Hpetsc->MakeDenseMatrix();
  const double kTolerance = std::numeric_limits<double>::epsilon() * H.rows();
  EXPECT_TRUE(CompareMatrices(dense_from_petsc, dense_from_pentadiagonal,
                              kTolerance, MatrixCompareType::relative));

  // Solve with Petsc's direct solver.
  VectorXd x_petsc_chol = b;
  start = steady_clock::now();
  PetscSolverStatus status = Hpetsc->Solve(
      PetscSymmetricBlockSparseMatrix::SolverType::kDirect,
      PetscSymmetricBlockSparseMatrix::PreconditionerType::kCholesky, b,
      &x_petsc_chol);
  end = steady_clock::now();
  wall_clock_time = std::chrono::duration<double>(end - start).count();
  fmt::print("Petsc Chol.   Wall clock: {:.4g} seconds. error: {}\n",
             wall_clock_time, (x_petsc_chol - x_gt).norm() / x_gt.norm());
  EXPECT_EQ(status, PetscSolverStatus::kSuccess);

  VectorXd x_petsc_minres = b;
  start = steady_clock::now();
  status = Hpetsc->Solve(
      PetscSymmetricBlockSparseMatrix::SolverType::kMINRES,
      PetscSymmetricBlockSparseMatrix::PreconditionerType::kIncompleteCholesky,
      b, &x_petsc_minres);
  end = steady_clock::now();
  wall_clock_time = std::chrono::duration<double>(end - start).count();
  fmt::print("Petsc MinRes. Wall clock: {:.4g} seconds. error: {}\n",
             wall_clock_time, (x_petsc_minres - x_gt).norm() / x_gt.norm());
  EXPECT_EQ(status, PetscSolverStatus::kSuccess);

  VectorXd x_petsc_cg = b;
  start = steady_clock::now();
  status = Hpetsc->Solve(
      PetscSymmetricBlockSparseMatrix::SolverType::kConjugateGradient,
      PetscSymmetricBlockSparseMatrix::PreconditionerType::kIncompleteCholesky,
      b, &x_petsc_cg);
  end = steady_clock::now();
  wall_clock_time = std::chrono::duration<double>(end - start).count();
  fmt::print("Petsc CG.     Wall clock: {:.4g} seconds. error: {}\n",
             wall_clock_time, (x_petsc_cg - x_gt).norm() / x_gt.norm());
  EXPECT_EQ(status, PetscSolverStatus::kSuccess);

  // const double kTolerance = std::numeric_limits<double>::epsilon() * size;
  // EXPECT_TRUE(
  //     CompareMatrices(x, x_expected, kTolerance,
  //     MatrixCompareType::relative));
}

// Solve H*x = b, where H has a high condition number
GTEST_TEST(PentaDiagonalMatrixTest, ConditionNumber) {
  const int block_size = 5;
  const int num_blocks = 30;
  const int size = num_blocks * block_size;


  std::cout << "Condition number, dense error, sparse error" << std::endl;
  for (double scale_factor = 1e1; scale_factor < 1e20; scale_factor *= 10) {
    // Generate a matrix H 
    const MatrixXd A = 1e4 * MatrixXd::Random(size, size);
    const MatrixXd P = MatrixXd::Identity(size, size) + A.transpose() * A;
    PentaDiagonalMatrix<double> H =
        PentaDiagonalMatrix<double>::MakeSymmetricFromLowerDense(P, num_blocks,
                                                                block_size);
    MatrixXd Hdense = H.MakeDense();

    // Modify H so it has the desired condition number
    Eigen::JacobiSVD<MatrixXd> svd(Hdense,
                                  Eigen::ComputeThinU | Eigen::ComputeThinV);
    const MatrixXd U = svd.matrixU();
    const MatrixXd V = svd.matrixV();
    VectorXd S = svd.singularValues();
    const double S_0 = S(0);
    const double S_end = S(size - 1);
    S = S_0 *
        (VectorXd::Ones(size) -
        ((scale_factor - 1) / scale_factor) * (S_0 * VectorXd::Ones(size) - S) / (S_0 - S_end));

    const MatrixXd H_reconstructed = U * S.asDiagonal() * V.transpose();
    H = PentaDiagonalMatrix<double>::MakeSymmetricFromLowerDense(
        H_reconstructed, num_blocks, block_size);
    Hdense = H.MakeDense();

    // Define a ground truth solution x
    const VectorXd x_gt = VectorXd::Random(size);

    // Define the vector b
    const VectorXd b = Hdense * x_gt;

    // Compute x using the Thomas algorithm (sparse)
    PentaDiagonalFactorization Hlu(H);
    EXPECT_EQ(Hlu.status(), PentaDiagonalFactorizationStatus::kSuccess);
    VectorXd x_sparse = b;
    Hlu.SolveInPlace(&x_sparse);

    // Compute x using LDLT (dense)
    const VectorXd x_dense = Hdense.ldlt().solve(b);

    // Compare with ground truth
    const double cond = 1 / Hdense.ldlt().rcond();
    const double dense_error = (x_gt - x_dense).norm();
    const double sparse_error = (x_gt - x_sparse).norm();
    std::cout << fmt::format("{}, {}, {}\n", cond, dense_error, sparse_error);

    const auto ldlt = Hdense.ldlt();    
    std::cout << "Dmin(ldlt): " << ldlt.vectorD().transpose().minCoeff() << std::endl;
    std::cout << "Dmax(ldlt): " << ldlt.vectorD().transpose().maxCoeff() << std::endl;
  }


}

}  // namespace internal
}  // namespace traj_opt
}  // namespace drake
