#include "drake/examples/sycl/simple_mesh_example/simple_mesh.h"

#include <sycl/sycl.hpp>

namespace drake {

// Constructor taking pre-allocated USM memory pointers
template <typename VectorType>
SimpleMesh<VectorType>::SimpleMesh(VectorType* p_MV, int* elements,
                                   size_t num_points, size_t num_elements) {
  p_MV_ = p_MV;
  elements_ = elements;
  num_points_ = num_points;
  num_elements_ = num_elements;
}

// No destructor needed - memory management is handled by the user

template <typename VectorType>
SimpleMesh<VectorType>::SimpleMesh(const SimpleMesh& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
}

template <typename VectorType>
SimpleMesh<VectorType>::SimpleMesh(SimpleMesh&& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
}

template <typename VectorType>
SimpleMesh<VectorType>& SimpleMesh<VectorType>::operator=(
    const SimpleMesh& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
  return *this;
}

template <typename VectorType>
SimpleMesh<VectorType>& SimpleMesh<VectorType>::operator=(SimpleMesh&& other) {
  p_MV_ = other.p_MV_;
  elements_ = other.elements_;
  num_points_ = other.num_points_;
  num_elements_ = other.num_elements_;
  return *this;
}

// Explicit instantiations
template class SimpleMesh<Vector3<double>>;
template class SimpleMesh<sycl::vec<double, 3>>;

}  // namespace drake
