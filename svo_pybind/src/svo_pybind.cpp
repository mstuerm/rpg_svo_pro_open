#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/transformation.h>
#include <svo/frame_handler_mono.h>
#include <svo/initialization.h>
#include <svo/reprojector.h>
#include <svo/direct/depth_filter.h>
#include <svo/direct/feature_detection_types.h>
#include <svo/tracker/feature_tracker.h>

namespace py = pybind11;

namespace {

template <typename T>
T value(const YAML::Node& config, const char* key, const T& fallback)
{
  return config[key] ? config[key].as<T>() : fallback;
}

svo::BaseOptions loadBaseOptions(const YAML::Node& config)
{
  svo::BaseOptions o;
  o.max_n_kfs = value(config, "max_n_kfs", static_cast<std::size_t>(5));
  o.quality_min_fts = value(config, "quality_min_fts", static_cast<std::size_t>(50));
  o.quality_max_fts_drop = value(config, "quality_max_drop_fts", 40);
  o.relocalization_max_trials = value(config, "relocalization_max_trials", static_cast<std::size_t>(50));
  o.poseoptim_prior_lambda = value(config, "poseoptim_prior_lambda", 0.0);
  o.poseoptim_using_unit_sphere = value(config, "poseoptim_using_unit_sphere", false);
  o.img_align_prior_lambda_rot = value(config, "img_align_prior_lambda_rot", 0.0);
  o.img_align_prior_lambda_trans = value(config, "img_align_prior_lambda_trans", 0.0);
  o.structure_optimization_max_pts = value(config, "structure_optimization_max_pts", 20);
  o.init_map_scale = value(config, "map_scale", 1.0);
  const std::string criterion = value(config, "kfselect_criterion", std::string("FORWARD"));
  o.kfselect_criterion = criterion == "FORWARD"
      ? svo::KeyframeCriterion::FORWARD : svo::KeyframeCriterion::DOWNLOOKING;
  o.kfselect_min_dist = value(config, "kfselect_min_dist", 0.12);
  o.kfselect_numkfs_upper_thresh = value(config, "kfselect_numkfs_upper_thresh", static_cast<std::size_t>(120));
  o.kfselect_numkfs_lower_thresh = value(config, "kfselect_numkfs_lower_thresh", static_cast<std::size_t>(70));
  o.kfselect_min_dist_metric = value(config, "kfselect_min_dist_metric", 0.01);
  o.kfselect_min_angle = value(config, "kfselect_min_angle", 20.0);
  o.kfselect_min_disparity = value(config, "kfselect_min_disparity", 40.0);
  o.kfselect_min_num_frames_between_kfs = value(config, "kfselect_min_num_frames_between_kfs", 2);
  o.img_align_max_level = value(config, "img_align_max_level", static_cast<std::size_t>(4));
  o.img_align_min_level = value(config, "img_align_min_level", static_cast<std::size_t>(2));
  o.img_align_robustification = value(config, "img_align_robustification", false);
  o.img_align_use_distortion_jacobian = value(config, "img_align_use_distortion_jacobian", false);
  o.img_align_est_illumination_gain = value(config, "img_align_est_illumination_gain", false);
  o.img_align_est_illumination_offset = value(config, "img_align_est_illumination_offset", false);
  o.poseoptim_thresh = value(config, "poseoptim_thresh", 2.0);
  o.update_seeds_with_old_keyframes = value(config, "update_seeds_with_old_keyframes", true);
  o.use_async_reprojectors = value(config, "use_async_reprojectors", false);
  o.trace_statistics = value(config, "trace_statistics", false);
  o.trace_dir = value(config, "trace_dir", std::string("/tmp/svo_pybind"));
  return o;
}

svo::DetectorOptions loadDetectorOptions(const YAML::Node& config)
{
  svo::DetectorOptions o;
  o.cell_size = value(config, "grid_size", 35);
  o.max_level = value(config, "n_pyr_levels", 3) - 1;
  o.threshold_primary = value(config, "detector_threshold_primary", 10);
  o.threshold_secondary = value(config, "detector_threshold_secondary", 200);
  o.threshold_shitomasi = value(config, "detector_threshold_shitomasi", 100);
  o.detector_type = value(config, "use_edgelets", true)
      ? svo::DetectorType::kFastGrad : svo::DetectorType::kFast;
  return o;
}

svo::DepthFilterOptions loadDepthFilterOptions(const YAML::Node& config)
{
  svo::DepthFilterOptions o;
  o.max_search_level = value(config, "n_pyr_levels", 3) - 1;
  o.use_threaded_depthfilter = value(config, "use_threaded_depthfilter", true);
  o.seed_convergence_sigma2_thresh = value(config, "seed_convergence_sigma2_thresh", 200.0);
  o.mappoint_convergence_sigma2_thresh = value(config, "mappoint_convergence_sigma2_thresh", 500.0);
  o.scan_epi_unit_sphere = value(config, "scan_epi_unit_sphere", false);
  o.affine_est_offset = value(config, "depth_filter_affine_est_offset", true);
  o.affine_est_gain = value(config, "depth_filter_affine_est_gain", false);
  o.max_n_seeds_per_frame = static_cast<std::size_t>(
      value(config, "max_fts", 120.0) * value(config, "max_seeds_ratio", 3.0));
  o.max_map_seeds_per_frame = static_cast<std::size_t>(value(config, "max_map_fts", 120.0));
  o.extra_map_points = value(config, "depth_filter_extra_map_points", false);
  return o;
}

svo::InitializationOptions loadInitializationOptions(const YAML::Node& config)
{
  svo::InitializationOptions o;
  o.init_min_features = value(config, "init_min_features", static_cast<std::size_t>(100));
  o.init_min_tracked = value(config, "init_min_tracked", static_cast<std::size_t>(80));
  o.init_min_inliers = value(config, "init_min_inliers", static_cast<std::size_t>(70));
  o.init_min_disparity = value(config, "init_min_disparity", 40.0);
  o.init_min_features_factor = value(config, "init_min_features_factor", 2.0);
  o.reproj_error_thresh = value(config, "reproj_err_thresh", 2.0);
  o.init_disparity_pivot_ratio = value(config, "init_disparity_pivot_ratio", 0.5);
  const std::string method = value(config, "init_method", std::string("FivePoint"));
  if(method == "Homography") o.init_type = svo::InitializerType::kHomography;
  else if(method == "TwoPoint") o.init_type = svo::InitializerType::kTwoPoint;
  else if(method == "OneShot") o.init_type = svo::InitializerType::kOneShot;
  else o.init_type = svo::InitializerType::kFivePoint;
  return o;
}

svo::FeatureTrackerOptions loadTrackerOptions(const YAML::Node& config)
{
  svo::FeatureTrackerOptions o;
  o.klt_max_level = value(config, "klt_max_level", 4);
  o.klt_min_level = value(config, "klt_min_level", 0);
  return o;
}

svo::ReprojectorOptions loadReprojectorOptions(const YAML::Node& config)
{
  svo::ReprojectorOptions o;
  o.max_n_kfs = value(config, "reprojector_max_n_kfs", 5);
  o.max_n_features_per_frame = value(config, "max_fts", 160);
  o.cell_size = value(config, "grid_size", 35);
  o.reproject_unconverged_seeds = value(config, "reproject_unconverged_seeds", true);
  o.max_unconverged_seeds_ratio = value(config, "max_unconverged_seeds_ratio", -1.0);
  o.min_required_features = value(config, "quality_min_fts", 50);
  o.seed_sigma2_thresh = value(config, "seed_convergence_sigma2_thresh", 200.0);
  o.affine_est_offset = value(config, "reprojector_affine_est_offset", true);
  o.affine_est_gain = value(config, "reprojector_affine_est_gain", false);
  return o;
}

class SvoPybind
{
public:
  SvoPybind(const std::string& config_path, const std::string& calib_path)
  {
    const YAML::Node config = YAML::LoadFile(config_path);
    auto cameras = svo::CameraBundle::loadFromYaml(calib_path);
    if(cameras->numCameras() != 1)
      throw std::runtime_error("svo_pybind currently supports one camera");
    svo_ = std::make_shared<svo::FrameHandlerMono>(
        loadBaseOptions(config), loadDepthFilterOptions(config),
        loadDetectorOptions(config), loadInitializationOptions(config),
        loadReprojectorOptions(config), loadTrackerOptions(config), cameras);
    svo_->start();
  }

  py::tuple step(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
                 const int64_t timestamp_ns)
  {
    return process(image, nullptr, timestamp_ns);
  }

  py::tuple step_with_depth(
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
      const py::array_t<float, py::array::c_style | py::array::forcecast>& depth,
      const int64_t timestamp_ns)
  {
    return process(image, &depth, timestamp_ns);
  }

  void reset()
  {
    svo_->reset();
    svo_->start();
  }

private:
  py::tuple process(
      const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image,
      const py::array_t<float, py::array::c_style | py::array::forcecast>* depth,
      const int64_t timestamp_ns)
  {
    const auto image_info = image.request();
    if(image_info.ndim != 2)
      throw std::runtime_error("image must have shape (height, width)");
    cv::Mat image_mat(static_cast<int>(image_info.shape[0]),
                      static_cast<int>(image_info.shape[1]), CV_8UC1,
                      image_info.ptr);

    std::unique_ptr<cv::Mat> depth_mat;
    if(depth != nullptr)
    {
      const auto depth_info = depth->request();
      if(depth_info.ndim != 2 || depth_info.shape[0] != image_info.shape[0] ||
         depth_info.shape[1] != image_info.shape[1])
        throw std::runtime_error("depth must have the same 2D shape as image");
      depth_mat.reset(new cv::Mat(static_cast<int>(depth_info.shape[0]),
                                  static_cast<int>(depth_info.shape[1]), CV_32FC1,
                                  depth_info.ptr));
      svo_->setDepthPriorImage(*depth_mat);
    }

    const auto t0 = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      svo_->addImageBundle({image_mat}, static_cast<uint64_t>(timestamp_ns));
    }
    const auto t1 = std::chrono::steady_clock::now();

    py::array_t<double> pose(16);
    auto pose_view = pose.mutable_unchecked<1>();
    for(ssize_t i = 0; i < 16; ++i) pose_view(i) = std::numeric_limits<double>::quiet_NaN();
    const auto frames = svo_->getLastFrames();
    if(frames != nullptr)
    {
      const Eigen::Matrix<double, 4, 4> matrix = frames->at(0)->T_world_cam().getTransformationMatrix();
      std::memcpy(pose.mutable_data(), matrix.data(), 16 * sizeof(double));
    }
    const double runtime_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return py::make_tuple(pose, static_cast<int>(svo_->stage()), runtime_us);
  }

  std::shared_ptr<svo::FrameHandlerMono> svo_;
};

}  // namespace

PYBIND11_MODULE(svo_pybind, module)
{
  module.doc() = "Non-ROS pybind11 wrapper around upstream SVO";
  py::class_<SvoPybind>(module, "SVO")
      .def(py::init<const std::string&, const std::string&>())
      .def("step", &SvoPybind::step)
      .def("step_with_depth", &SvoPybind::step_with_depth)
      .def("reset", &SvoPybind::reset);
}
