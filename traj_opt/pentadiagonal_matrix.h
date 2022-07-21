#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * A sparse representation of a symetric penta-diagonal matrix. 
 * 
 * A penta-diagonal matrix is of the form
 * 
 *   [ C0 D0 E0  0  0  0  0  0 ]
 *   [ B1 C1 D1 E1  0  0  0  0 ]
 *   [ A2 B2 C2 D2 E2  0  0  0 ]
 *   [  0 A3 B3 C3 D3 E3  0  0 ]
 *                 ...
 *                    ...
 *   [  0  0  0  0  0 AN BN CN ]
 * 
 * In our case this is a symmetric matrix, so we only need to
 * store Ci, Di, and Ei.
 */
class SymmetricPentadiagonalMatrix {
 public:
  // Allocate a symmetric pentadiagonal matrix with N blocks on the diagonal 
  // and each block being an (n x n) matrix. 
  SymmetricPentadiagonalMatrix(int N, int n);

  // Sets the value of the i^th diagonal block.
  // TODO(vincekurtz): return a mutable Eigen::Ref instead?
  void SetDiagonalBlock(int i, MatrixXd Ci);
  
  // Sets the value of the i^th block on the first diagonal.
  void SetFirstDiagonalBlock(int i, MatrixXd Di);

  // Sets the value of the i^th block on the second diagonal.
  void SetSecondDiagonalBlock(int i, MatrixXd Ei);

  // Solve
  //    H * x = b,
  // for x, where H is this matrix, and b is a given vector.
  //
  // Implements the algorithm described in
  //     Benkert et al., "An Efficient Implementation of the Thomas-Algorithm
  //     for Block Penta-diagonal Systems on Vector Computers", ICCS 2007.
  void SolveThomas(const MatrixXd& b, EigenPtr<VectorXd> x);

  // Solve
  //    H * x = b,
  // for x, where H is this matrix, and b is a given vector.
  //
  // Does so by first constructing a dense matrix, so this will be super slow. 
  void SolveDense(const MatrixXd& b, EigenPtr<VectorXd> x);

 private:
  // Vector of diagonal blocks [C0, C1, ..., CN].
  std::vector<MatrixXd> diagonal_;

  // Vector of blocks on the first diagonal [D0, D1, ..., DN].
  // DN is constrained to be zero.
  std::vector<MatrixXd> first_diagonal_;

  // Vector of blocks on the second diagonal [E0, E1, ..., EN].
  // EN and E(N-1) are constrained to be zero.
  std::vector<MatrixXd> second_first_diagonal_;
};

}  // namespace traj_opt
}  // namespace drake
