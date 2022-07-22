#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace trajopt {
namespace internal {

class PentaDiagonalMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PentaDiagonalMatrix);

  PentaDiagonalMatrix(std::vector<Eigen::MatrixXd> A,
                      std::vector<Eigen::MatrixXd> B,
                      std::vector<Eigen::MatrixXd> C,
                      std::vector<Eigen::MatrixXd> D,
                      std::vector<Eigen::MatrixXd> E);

  // Constructor for a symmetric penta-diagonal matrix.
  // That is, E = A and D = B. C must be symmetric.  
  PentaDiagonalMatrix(std::vector<Eigen::MatrixXd> A,
                      std::vector<Eigen::MatrixXd> B,
                      std::vector<Eigen::MatrixXd> C);

  static PentaDiagonalMatrix MakeIdentity(int num_blocks, int block_size);

  static PentaDiagonalMatrix MakeSymmetricFromLowerDense(
      const Eigen::MatrixXd& M, int num_blocks, int block_size);

  Eigen::MatrixXd MakeDense() const;      

  // The size k of each of the blocks in the diagonals. All blocks have the same
  // size k x k.
  int block_size() const { return A_.size() == 0 ? 0 : A_[0].rows(); }

  // Returns the the total number of rows.  
  int rows() const { return block_rows() * block_size(); }

  // Returns the number of block rows.
  int block_rows() const { return C_.size(); }

  int block_cols() const { return block_rows(); }

  int cols() const { return rows(); }

  // Returns a reference to the second lower diagonal.  
  const std::vector<Eigen::MatrixXd>& A() const { return A_; }

  // Returns a reference to the first lower diagonal.
  const std::vector<Eigen::MatrixXd>& B() const { return B_; }

  // Returns a reference to the main diagonal.
  const std::vector<Eigen::MatrixXd>& C() const { return C_; }

  // Returns a reference to the first upper diagonal.
  const std::vector<Eigen::MatrixXd>& D() const { return D_; }

  // Returns a reference to the second upper diagonal.
  const std::vector<Eigen::MatrixXd>& E() const { return E_; }

  // TODO: methods to build from/to dense matrices.

 private:
  static bool VerifyAllBlocksOfSameSize(const std::vector<Eigen::MatrixXd>& X,
                                        int size);
  bool VerifySizes() const;

  std::vector<Eigen::MatrixXd> A_;
  std::vector<Eigen::MatrixXd> B_;
  std::vector<Eigen::MatrixXd> C_;
  std::vector<Eigen::MatrixXd> D_;
  std::vector<Eigen::MatrixXd> E_;
};

}  // namespace internal
}  // namespace trajopt
}  // namespace multibody
}  // namespace drake