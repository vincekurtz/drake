#include <chrono>
#include <iostream>


#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>

using namespace drake;

/**
 * A normal function that computes y = f(x)
 */
template <typename T>
void TestFunction(const MatrixX<double>& A, const VectorX<T>& x,
                  VectorX<T>* y) {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == x.size());
  y->noalias() = x.transpose() * A * x;
}

// Same as TestFunction, but optimized (via specialization) for T = AutoDiffXd.
template <typename T>
void TestFunctionOptimized(const MatrixX<double>& A, const VectorX<T>& x,
                           VectorX<T>* y) {
  TestFunction(A, x, y);
}

/**
 * Specialize `TestFunctionOptimized()` for T = AutoDiffXd so that calling code
 * does not need to worry about wether it works with <double> or AutoDiffXd.
 * This uses our special insights to compute dy/dx analytically. These are then
 * loaded back into the derivatives() of y.
 */
template <>
void TestFunctionOptimized(const MatrixX<double>& A,
                           const VectorX<AutoDiffXd>& x,
                           VectorX<AutoDiffXd>* y) {
  DRAKE_DEMAND(y != nullptr);
  DRAKE_DEMAND(y->size() == x.size());

  VectorX<double> val = math::ExtractValue(x);
  MatrixX<double> grad = math::ExtractGradient(x);

  // Compute the main result using floating point math
  VectorX<double> res(x.size());
  TestFunction(A, val, &res);

  // Compute the derivatives() analytically.
  const MatrixX<double> dy_dx = val.transpose() * (A + A.transpose());
  const MatrixX<double> deriv = dy_dx * grad;  // chain rule

  // Load those into a newly computed result.
  // Note that other versions of InitializeAutoDiff may end up being this
  // more useful to us, such as the one that takes a pointer to y
  // (the resulting autodiff matrix) rather than returning it.
  math::InitializeAutoDiff(res, deriv, y);
}

int main() {
  // Size of the vector x
  int n = 1000;

  // Define original vector
  VectorX<double> x(n);
  x.setRandom(n, 1);
  const VectorX<AutoDiffXd> x_ad = math::InitializeAutoDiff(x);

  // Define the matrix that we use for the test function y = x'Ax
  MatrixX<double> A(n, n);
  A.setRandom(n, n);

  // Set up some timers
  auto st = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed_normal;
  std::chrono::duration<float> elapsed_fancy;

  // Do some maths
  st = std::chrono::high_resolution_clock::now();
  VectorX<AutoDiffXd> y_normal(n);
  TestFunction(A, x_ad, &y_normal);
  elapsed_normal = std::chrono::high_resolution_clock::now() - st;

  st = std::chrono::high_resolution_clock::now();
  VectorX<AutoDiffXd> y_fancy(n);
  TestFunctionOptimized(A, x_ad, &y_fancy);
  elapsed_fancy = std::chrono::high_resolution_clock::now() - st;

  // Print the results
  std::cout << "normal method: " << elapsed_normal.count() << " seconds"
            << std::endl;
  std::cout << "fancy method: " << elapsed_fancy.count() << " seconds"
            << std::endl;

  std::cout << std::endl;

  // Sanity check
  const VectorX<double> value_diff =
      math::ExtractValue(y_normal) - math::ExtractValue(y_fancy);
  const MatrixX<double> deriv_diff =
      math::ExtractGradient(y_normal) - math::ExtractGradient(y_fancy);

  std::cout << fmt::format("Values error: {}\n", value_diff.norm());
  std::cout << fmt::format("Gradients error: {}\n", deriv_diff.norm());

  return 0;
}
