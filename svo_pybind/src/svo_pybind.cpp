#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>
#include <ros/time.h>

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
#include <svo/imu_handler.h>
#include <svo/initialization.h>
#include <svo/reprojector.h>
#include <svo/ceres_backend_interface.hpp>
#include <svo/motion_detector.hpp>
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
  o.backend_scale_stable_thresh = value(config, "backend_scale_stable_thresh", 0.02);
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

svo::CeresBackendOptions loadCeresBackendOptions(const YAML::Node& config)
{
  svo::CeresBackendOptions o;
  o.verbose = value(config, "ceres_verbose", false);
  o.marginalize = value(config, "ceres_marginalize", true);
  o.num_iterations = static_cast<std::size_t>(value(config, "ceres_num_iterations", 3));
  o.num_threads = static_cast<std::size_t>(value(config, "ceres_num_threads", 2));
  o.num_imu_frames = static_cast<std::size_t>(value(config, "ceres_num_imu_frames", 3));
  o.num_keyframes = static_cast<std::size_t>(value(config, "max_n_kfs", 5));
  o.max_iteration_time = value(config, "ceres_max_iteration_time", -1.0);
  o.remove_marginalization_term_after_correction_ =
      value(config, "ceres_remove_marginalization_term_after_loop", false);
  o.recalculate_imu_terms_after_loop =
      value(config, "ceres_recalculate_imu_terms_after_loop", false);
  o.remove_fixation_min_num_fixed_landmarks_ = static_cast<std::size_t>(
      value(config, "ceres_remove_fixation_min_num_fixed_points", 10));
  o.max_fixed_lm_in_ceres_ = static_cast<std::size_t>(
      value(config, "ceres_max_fixed_landmarks", 50));
  return o;
}

svo::CeresBackendInterfaceOptions loadCeresInterfaceOptions(
    const YAML::Node& config)
{
  svo::CeresBackendInterfaceOptions o;
  o.min_num_obs = static_cast<std::size_t>(value(config, "backend_min_num_obs", 2));
  o.min_parallax_thresh = value(config, "backend_min_parallax_deg", 2.0) * M_PI / 180.0;
  o.only_use_corners = value(config, "backend_only_use_corners", false);
  o.use_zero_motion_detection = value(config, "backend_use_zero_motion_detection", true);
  o.backend_zero_motion_check_n_frames = static_cast<std::size_t>(
      value(config, "backend_zero_motion_check_n_frames", 5));
  o.use_outlier_rejection = value(config, "use_outlier_rejection", true);
  o.outlier_rejection_px_threshold = value(config, "outlier_rejection_px_threshold", 2.0);
  o.skip_optimization_when_tracking_bad =
      value(config, "skip_optimization_when_tracking_bad", false);
  o.min_added_measurements = static_cast<std::size_t>(
      value(config, "skip_optim_min_obs", 20));
  o.refine_extrinsics = value(config, "backend_refine_extrinsics", false);
  o.extrinsics_pos_sigma_meter =
      value(config, "backend_extrinsics_pos_sigma_meter", 0.05);
  o.extrinsics_rot_sigma_rad =
      value(config, "backend_extrinsics_rot_sigma_deg", 5.0) * M_PI / 180.0;
  return o;
}

svo::MotionDetectorOptions loadMotionDetectorOptions(const YAML::Node& config)
{
  svo::MotionDetectorOptions o;
  o.px_diff_threshold = value(config, "zero_motion_px_diff_threshold", 0.5);
  o.ratio_moving_pixels_threshold =
      value(config, "zero_motion_ratio_moving_pixels_threshold", 0.1);
  o.min_number_correspondences =
      value(config, "zero_motion_min_number_correspondences", 5);
  o.max_features_to_check = value(config, "zero_motion_max_features_to_check", 100);
  o.sigma = value(config, "zero_motion_sigma", 0.05);
  return o;
}

class SvoPybind
{
public:
  SvoPybind(const std::string& config_path, const std::string& calib_path)
  {
    // SVO Pro is compiled with its ROS timing adapter and some backend paths
    // use ros::Time as a wall-clock abstraction. Standalone pybind does not
    // construct a NodeHandle, so initialize only the time subsystem. This
    // starts no ROS node, spinner, publisher, subscriber, or communication.
    ros::Time::init();
    const YAML::Node config = YAML::LoadFile(config_path);
    auto cameras = svo::CameraBundle::loadFromYaml(calib_path);
    if(cameras->numCameras() != 1)
      throw std::runtime_error("svo_pybind currently supports one camera");
    svo_ = std::make_shared<svo::FrameHandlerMono>(
        loadBaseOptions(config), loadDepthFilterOptions(config),
        loadDetectorOptions(config), loadInitializationOptions(config),
        loadReprojectorOptions(config), loadTrackerOptions(config), cameras);

    if(value(config, "use_imu", false))
    {
      const svo::ImuCalibration imu_calibration =
          svo::ImuHandler::loadCalibrationFromFile(calib_path);
      const svo::ImuInitialization imu_initialization =
          svo::ImuHandler::loadInitializationFromFile(calib_path);
      svo::IMUHandlerOptions imu_options;
      imu_options.temporal_stationary_check =
          value(config, "imu_temporal_stationary_check", false);
      imu_options.temporal_window_length_sec_ =
          value(config, "imu_temporal_window_length_sec", 0.5);
      imu_options.stationary_acc_sigma_thresh_ =
          value(config, "stationary_acc_sigma_thresh", 0.0);
      imu_options.stationary_gyr_sigma_thresh_ =
          value(config, "stationary_gyr_sigma_thresh", 0.0);
      imu_handler_ = std::make_shared<svo::ImuHandler>(
          imu_calibration, imu_initialization, imu_options);
      svo_->imu_handler_ = imu_handler_;

      if(value(config, "use_ceres_backend", false))
      {
        ceres_backend_ = std::make_shared<svo::CeresBackendInterface>(
            loadCeresInterfaceOptions(config), loadCeresBackendOptions(config),
            loadMotionDetectorOptions(config), cameras);
        ceres_backend_->setImu(imu_handler_);
        svo_->setBundleAdjuster(ceres_backend_);
        start_backend_thread_after_first_frame_ =
            value(config, "ba_parallelized", true);
      }
    }
    svo_->start();
  }

  void add_imu(
      const int64_t timestamp_ns,
      const py::array_t<double, py::array::c_style | py::array::forcecast>& acceleration,
      const py::array_t<double, py::array::c_style | py::array::forcecast>& angular_velocity)
  {
    if(!imu_handler_)
      throw std::runtime_error("SVO was constructed with use_imu=false");
    const auto acc = acceleration.request();
    const auto gyro = angular_velocity.request();
    if(acc.ndim != 1 || acc.shape[0] != 3 || gyro.ndim != 1 || gyro.shape[0] != 3)
      throw std::runtime_error("acceleration and angular_velocity must have shape (3,)");
    if(timestamp_ns <= latest_imu_timestamp_ns_)
      throw std::runtime_error("IMU timestamps must be strictly increasing");
    const double* acc_data = static_cast<const double*>(acc.ptr);
    const double* gyro_data = static_cast<const double*>(gyro.ptr);
    const Eigen::Vector3d acc_vector(acc_data[0], acc_data[1], acc_data[2]);
    const Eigen::Vector3d gyro_vector(gyro_data[0], gyro_data[1], gyro_data[2]);
    if(!acc_vector.allFinite() || !gyro_vector.allFinite())
      throw std::runtime_error("IMU measurement contains non-finite values");
    imu_handler_->addImuMeasurement(svo::ImuMeasurement(
        static_cast<double>(timestamp_ns) * 1e-9, gyro_vector, acc_vector));
    latest_imu_timestamp_ns_ = timestamp_ns;
    ++num_imu_measurements_;
  }

  std::size_t num_imu_measurements() const { return num_imu_measurements_; }
  bool has_ceres_backend() const { return static_cast<bool>(ceres_backend_); }

  py::tuple latest_speed_bias() const
  {
    py::array_t<double> result(9);
    auto result_view = result.mutable_unchecked<1>();
    for(ssize_t i = 0; i < 9; ++i)
      result_view(i) = std::numeric_limits<double>::quiet_NaN();
    double timestamp = std::numeric_limits<double>::quiet_NaN();
    if(ceres_backend_ && ceres_backend_->getNumFrames() > 0)
    {
      Eigen::Matrix<double, 9, 1> speed_bias;
      svo::Transformation T_WS;
      ceres_backend_->getLatestSpeedBiasPose(&speed_bias, &T_WS, &timestamp);
      std::memcpy(result.mutable_data(), speed_bias.data(), 9 * sizeof(double));
    }
    return py::make_tuple(result, timestamp);
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
      if(imu_handler_ && !ceres_backend_)
      {
        const double timestamp_sec = static_cast<double>(timestamp_ns) * 1e-9;
        if(!svo_->hasStarted())
        {
          svo::Quaternion R_imu_world;
          if(!imu_handler_->getInitialAttitude(timestamp_sec, R_imu_world))
            throw std::runtime_error("Could not initialize frontend IMU attitude");
          svo_->setRotationPrior(R_imu_world);
        }
        else if(const auto last_frames = svo_->getLastFrames())
        {
          svo::Quaternion R_lastimu_newimu;
          if(imu_handler_->getRelativeRotationPrior(
                 last_frames->getMinTimestampSeconds(), timestamp_sec, false,
                 R_lastimu_newimu))
          {
            svo_->setRotationIncrementPrior(R_lastimu_newimu);
          }
        }
      }
      svo_->addImageBundle({image_mat}, static_cast<uint64_t>(timestamp_ns));
      if(start_backend_thread_after_first_frame_ && !backend_thread_started_ &&
         svo_->stage() == svo::Stage::kTracking)
      {
        ceres_backend_->startThread();
        backend_thread_started_ = true;
      }
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
  std::shared_ptr<svo::ImuHandler> imu_handler_;
  std::shared_ptr<svo::CeresBackendInterface> ceres_backend_;
  bool start_backend_thread_after_first_frame_ = false;
  bool backend_thread_started_ = false;
  int64_t latest_imu_timestamp_ns_ = -1;
  std::size_t num_imu_measurements_ = 0;
};

}  // namespace

PYBIND11_MODULE(svo_pybind, module)
{
  module.doc() = "Non-ROS pybind11 wrapper around upstream SVO";
  py::class_<SvoPybind>(module, "SVO")
      .def(py::init<const std::string&, const std::string&>())
      .def("step", &SvoPybind::step)
      .def("step_with_depth", &SvoPybind::step_with_depth)
      .def("add_imu", &SvoPybind::add_imu)
      .def_property_readonly("num_imu_measurements", &SvoPybind::num_imu_measurements)
      .def_property_readonly("has_ceres_backend", &SvoPybind::has_ceres_backend)
      .def("latest_speed_bias", &SvoPybind::latest_speed_bias)
      .def("reset", &SvoPybind::reset);
}
