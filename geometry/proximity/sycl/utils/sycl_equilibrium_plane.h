#pragma once

#include <cmath>

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

/* Computes the equilibrium plane between two pressure fields.
 * The equilibrium plane is defined as the plane where the pressure
 * values of both fields are equal.
 *
 * @param gradP_A_Wo_x X component of pressure gradient for geometry A in world
 * frame
 * @param gradP_A_Wo_y Y component of pressure gradient for geometry A in world
 * frame
 * @param gradP_A_Wo_z Z component of pressure gradient for geometry A in world
 * frame
 * @param p_A_Wo Pressure at world origin for geometry A
 * @param gradP_B_Wo_x X component of pressure gradient for geometry B in world
 * frame
 * @param gradP_B_Wo_y Y component of pressure gradient for geometry B in world
 * frame
 * @param gradP_B_Wo_z Z component of pressure gradient for geometry B in world
 * frame
 * @param p_B_Wo Pressure at world origin for geometry B
 * @param eq_plane_out Output array to store equilibrium plane data:
 *   [0-2]: normalized normal vector
 *   [3-5]: point on plane
 *   [6]: gM value (dot product of gradient A with normal)
 *   [7]: gN value (negative dot product of gradient B with normal)
 * @return true if a valid equilibrium plane was found, false otherwise
 */
SYCL_EXTERNAL inline bool ComputeEquilibriumPlane(
    double gradP_A_Wo_x, double gradP_A_Wo_y, double gradP_A_Wo_z,
    double p_A_Wo, double gradP_B_Wo_x, double gradP_B_Wo_y,
    double gradP_B_Wo_z, double p_B_Wo, double* eq_plane_out) {
  const double n_W_x = gradP_A_Wo_x - gradP_B_Wo_x;
  const double n_W_y = gradP_A_Wo_y - gradP_B_Wo_y;
  const double n_W_z = gradP_A_Wo_z - gradP_B_Wo_z;

  const double n_W_norm_sq = n_W_x * n_W_x + n_W_y * n_W_y + n_W_z * n_W_z;
  const double gradP_A_W_norm_sq = gradP_A_Wo_x * gradP_A_Wo_x +
                                   gradP_A_Wo_y * gradP_A_Wo_y +
                                   gradP_A_Wo_z * gradP_A_Wo_z;
  const double gradP_B_W_norm_sq = gradP_B_Wo_x * gradP_B_Wo_x +
                                   gradP_B_Wo_y * gradP_B_Wo_y +
                                   gradP_B_Wo_z * gradP_B_Wo_z;

  // Early exit if the normal is zero
  if (n_W_norm_sq <= 0.0) {
    return false;
  }

  // Pipeline reciprocal square roots
  const double n_W_inv_norm = sycl::rsqrt(n_W_norm_sq);
  const double gradP_A_W_inv_norm = sycl::rsqrt(gradP_A_W_norm_sq);
  const double gradP_B_W_inv_norm = sycl::rsqrt(gradP_B_W_norm_sq);

  // Conditional checks
  const double dot_n_gA =
      n_W_x * gradP_A_Wo_x + n_W_y * gradP_A_Wo_y + n_W_z * gradP_A_Wo_z;
  const double dot_n_gB =
      n_W_x * gradP_B_Wo_x + n_W_y * gradP_B_Wo_y + n_W_z * gradP_B_Wo_z;

  const double cos_theta_A = dot_n_gA * n_W_inv_norm * gradP_A_W_inv_norm;

  constexpr double kAlpha = 5.0 * M_PI / 8.0;
  const double kCosAlpha = sycl::cos(kAlpha);

  if (cos_theta_A <= kCosAlpha) {
    return false;
  }

  const double cos_theta_B = -dot_n_gB * n_W_inv_norm * gradP_B_W_inv_norm;

  if (cos_theta_B <= kCosAlpha) {
    return false;
  }

  const double n_W_x_normalized = n_W_x * n_W_inv_norm;
  const double n_W_y_normalized = n_W_y * n_W_inv_norm;
  const double n_W_z_normalized = n_W_z * n_W_inv_norm;

  const double gM = dot_n_gA * n_W_inv_norm;
  const double gN = -dot_n_gB * n_W_inv_norm;

  // Plane point calculation
  const double p_diff = p_B_Wo - p_A_Wo;
  const double n_W_norm_sq_rcp = n_W_inv_norm * n_W_inv_norm;
  const double scale = p_diff * n_W_norm_sq_rcp;

  const double p_WQ_x = scale * n_W_x;
  const double p_WQ_y = scale * n_W_y;
  const double p_WQ_z = scale * n_W_z;

  // Store final results
  eq_plane_out[0] = n_W_x_normalized;
  eq_plane_out[1] = n_W_y_normalized;
  eq_plane_out[2] = n_W_z_normalized;
  eq_plane_out[3] = p_WQ_x;
  eq_plane_out[4] = p_WQ_y;
  eq_plane_out[5] = p_WQ_z;
  eq_plane_out[6] = gM;
  eq_plane_out[7] = gN;

  return true;
}
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake