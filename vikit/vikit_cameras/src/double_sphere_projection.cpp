#include "vikit/cameras/double_sphere_projection.h"

#include <cmath>
#include <limits>

#include <glog/logging.h>

namespace vk {
namespace cameras {

DoubleSphereProjection::DoubleSphereProjection(
    const Eigen::VectorXd& intrinsics)
    : DoubleSphereProjection(
          intrinsics.size() == 6 ? intrinsics[0] : 0.0,
          intrinsics.size() == 6 ? intrinsics[1] : 0.0,
          intrinsics.size() == 6 ? intrinsics[2] : 0.0,
          intrinsics.size() == 6 ? intrinsics[3] : 0.0,
          intrinsics.size() == 6 ? intrinsics[4] : 0.0,
          intrinsics.size() == 6 ? intrinsics[5] : 0.0)
{
  CHECK_EQ(intrinsics.size(), 6)
      << "Double-sphere intrinsics must be [xi, alpha, fx, fy, cx, cy]";
}

DoubleSphereProjection::DoubleSphereProjection(
    double xi, double alpha, double fx, double fy, double cx, double cy)
    : xi_(xi), alpha_(alpha), fx_(fx), fy_(fy), cx_(cx), cy_(cy)
{
  CHECK(std::isfinite(xi_));
  CHECK_GT(alpha_, 0.0);
  CHECK_LT(alpha_, 1.0);
  CHECK_GT(fx_, 0.0);
  CHECK_GT(fy_, 0.0);
  CHECK(std::isfinite(cx_));
  CHECK(std::isfinite(cy_));
}

bool DoubleSphereProjection::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_bearing_vector) const
{
  CHECK_NOTNULL(out_bearing_vector);
  const double mx = (keypoint[0] - cx_) / fx_;
  const double my = (keypoint[1] - cy_) / fy_;
  const double r2 = mx * mx + my * my;
  const double root_argument = 1.0 - (2.0 * alpha_ - 1.0) * r2;
  if(root_argument < 0.0)
    return false;

  const double denominator =
      alpha_ * std::sqrt(root_argument) + 1.0 - alpha_;
  if(std::abs(denominator) < 1e-12)
    return false;
  const double mz = (1.0 - alpha_ * alpha_ * r2) / denominator;
  const double second_root = mz * mz + (1.0 - xi_ * xi_) * r2;
  if(second_root < 0.0)
    return false;
  const double scale_denominator = mz * mz + r2;
  if(scale_denominator < 1e-12)
    return false;
  const double scale =
      (mz * xi_ + std::sqrt(second_root)) / scale_denominator;
  *out_bearing_vector << scale * mx, scale * my, scale * mz - xi_;
  out_bearing_vector->normalize();
  return out_bearing_vector->allFinite();
}

void DoubleSphereProjection::project3(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point) const
{
  CHECK_NOTNULL(out_keypoint);
  constexpr double kEpsilon = 1e-12;
  const double x = point_3d[0];
  const double y = point_3d[1];
  const double z = point_3d[2];
  const double d1 = point_3d.norm();
  if(d1 < kEpsilon)
  {
    out_keypoint->setConstant(std::numeric_limits<double>::quiet_NaN());
    if(out_jacobian_point) out_jacobian_point->setZero();
    return;
  }

  const double z_xi = z + xi_ * d1;
  const double d2 = std::sqrt(x * x + y * y + z_xi * z_xi);
  const double denominator = alpha_ * d2 + (1.0 - alpha_) * z_xi;
  const double w1 = alpha_ <= 0.5 ? alpha_ / (1.0 - alpha_)
                                  : (1.0 - alpha_) / alpha_;
  const double w2 = (w1 + xi_) /
      std::sqrt(2.0 * w1 * xi_ + xi_ * xi_ + 1.0);
  if(d2 < kEpsilon || denominator <= kEpsilon || z <= -w2 * d1)
  {
    out_keypoint->setConstant(std::numeric_limits<double>::quiet_NaN());
    if(out_jacobian_point) out_jacobian_point->setZero();
    return;
  }

  (*out_keypoint)[0] = fx_ * x / denominator + cx_;
  (*out_keypoint)[1] = fy_ * y / denominator + cy_;

  if(out_jacobian_point)
  {
    const Eigen::RowVector3d d_d1 = point_3d.transpose() / d1;
    Eigen::RowVector3d d_z_xi = xi_ * d_d1;
    d_z_xi[2] += 1.0;
    Eigen::RowVector3d d_d2;
    d_d2 << x, y, 0.0;
    d_d2 = (d_d2 + z_xi * d_z_xi) / d2;
    const Eigen::RowVector3d d_denominator =
        alpha_ * d_d2 + (1.0 - alpha_) * d_z_xi;
    const double denominator2 = denominator * denominator;
    out_jacobian_point->row(0) =
        fx_ * (Eigen::RowVector3d(1.0, 0.0, 0.0) * denominator
               - x * d_denominator) / denominator2;
    out_jacobian_point->row(1) =
        fy_ * (Eigen::RowVector3d(0.0, 1.0, 0.0) * denominator
               - y * d_denominator) / denominator2;
  }
}

double DoubleSphereProjection::errorMultiplier() const
{
  return 0.5 * (fx_ + fy_);
}

double DoubleSphereProjection::getAngleError(double img_err) const
{
  Eigen::Vector3d center;
  Eigen::Vector3d offset_x;
  Eigen::Vector3d offset_y;
  if(!backProject3(Eigen::Vector2d(cx_, cy_), &center) ||
     !backProject3(Eigen::Vector2d(cx_ + img_err, cy_), &offset_x) ||
     !backProject3(Eigen::Vector2d(cx_, cy_ + img_err), &offset_y))
    return img_err / errorMultiplier();
  const double dot_x = std::max(-1.0, std::min(1.0, center.dot(offset_x)));
  const double dot_y = std::max(-1.0, std::min(1.0, center.dot(offset_y)));
  return 0.5 * (std::acos(dot_x) + std::acos(dot_y));
}

void DoubleSphereProjection::print(std::ostream& out) const
{
  out << "  Projection = Double Sphere" << std::endl;
  out << "  xi, alpha = " << xi_ << ", " << alpha_ << std::endl;
  out << "  focal length = " << fx_ << ", " << fy_ << std::endl;
  out << "  principal point = " << cx_ << ", " << cy_ << std::endl;
}

Eigen::VectorXd DoubleSphereProjection::getIntrinsicParameters() const
{
  Eigen::VectorXd intrinsics(6);
  intrinsics << xi_, alpha_, fx_, fy_, cx_, cy_;
  return intrinsics;
}

Eigen::VectorXd DoubleSphereProjection::getDistortionParameters() const
{
  return Eigen::VectorXd();
}

}  // namespace cameras
}  // namespace vk
