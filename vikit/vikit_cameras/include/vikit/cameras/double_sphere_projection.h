#pragma once

#include <Eigen/Core>

#include <vikit/cameras/camera_geometry_base.h>

namespace vk {
namespace cameras {

/// Usenko et al. double-sphere projection used by Kalibr's `ds` model.
/// Intrinsics are ordered [xi, alpha, fx, fy, cx, cy], matching Kalibr and
/// Habitat-Sim's FisheyeSensorDoubleSphereSpec.
class DoubleSphereProjection
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  const CameraGeometryBase::Type cam_type_ =
      CameraGeometryBase::Type::kDoubleSphere;

  explicit DoubleSphereProjection(const Eigen::VectorXd& intrinsics);
  DoubleSphereProjection(double xi, double alpha, double fx, double fy,
                         double cx, double cy);

  bool backProject3(
      const Eigen::Ref<const Eigen::Vector2d>& keypoint,
      Eigen::Vector3d* out_bearing_vector) const;

  void project3(
      const Eigen::Ref<const Eigen::Vector3d>& point_3d,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point) const;

  double errorMultiplier() const;
  double getAngleError(double img_err) const;
  void print(std::ostream& out) const;
  Eigen::VectorXd getIntrinsicParameters() const;
  Eigen::VectorXd getDistortionParameters() const;

 private:
  double xi_;
  double alpha_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
};

}  // namespace cameras
}  // namespace vk
