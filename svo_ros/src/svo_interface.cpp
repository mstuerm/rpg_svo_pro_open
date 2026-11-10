#include <svo_ros/svo_interface.h>

#include <ros/callback_queue.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <std_msgs/Float64MultiArray.h>

#include <unordered_map>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <svo_ros/svo_factory.h>
#include <svo_ros/visualizer.h>
#include <svo/common/frame.h>
#include <svo/map.h>
#include <svo/imu_handler.h>
#include <svo/common/camera.h>
#include <svo/common/conversions.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame_handler_stereo.h>
#include <svo/frame_handler_array.h>
#include <svo/initialization.h>
#include <svo/direct/depth_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <vikit/params_helper.h>
#include <vikit/timer.h>
#include <svo_ros/ceres_backend_factory.h>


#ifdef SVO_USE_GTSAM_BACKEND
#include <svo_ros/backend_factory.h>
#include <svo/backend/backend_interface.h>
#include <svo/backend/backend_optimizer.h>
#endif

#ifdef SVO_LOOP_CLOSING
#include <svo/online_loopclosing/loop_closing.h>
#endif

#ifdef SVO_GLOBAL_MAP
#include <svo/global_map.h>
#endif

namespace svo {

SvoInterface::SvoInterface(
    const PipelineType& pipeline_type,
    const ros::NodeHandle& nh,
    const ros::NodeHandle& private_nh)
  : nh_(nh)
  , pnh_(private_nh)
  , pipeline_type_(pipeline_type)
  , set_initial_attitude_from_gravity_(
      vk::param<bool>(pnh_, "set_initial_attitude_from_gravity", true))
  , automatic_reinitialization_(
      vk::param<bool>(pnh_, "automatic_reinitialization", false))
{
  trace_processing_stages_ = vk::param<bool>(pnh_, "trace_processing_stages", false);
  gravity_initialization_duration_ =
      vk::param<double>(pnh_, "gravity_initialization_duration", 0.0);
  CHECK(std::isfinite(gravity_initialization_duration_) &&
        gravity_initialization_duration_ >= 0.0);
  pub_gravity_initialization_ =
      pnh_.advertise<std_msgs::String>("gravity_initialization", 10, true);
  pub_remote_key_status_ = pnh_.advertise<std_msgs::String>("remote_key_status", 10, true);
  require_stationary_calibration_ =
      vk::param<bool>(pnh_, "require_stationary_calibration", false);
  stationary_calibration_duration_ =
      vk::param<double>(pnh_, "stationary_calibration_duration", 10.0);
  CHECK(std::isfinite(stationary_calibration_duration_) &&
        stationary_calibration_duration_ >= 1.0);
  pub_stationary_calibration_ =
      pnh_.advertise<std_msgs::String>("stationary_calibration", 10, true);
  publishStationaryCalibration("idle");
  stationary_calibration_timer_ = nh_.createTimer(ros::Duration(0.5),
      &SvoInterface::collectStationaryCalibration, this);
  if (trace_processing_stages_)
  {
    pub_processing_stage_ = pnh_.advertise<std_msgs::String>("processing_stage", 10, true);
    traceProcessingStage("waiting_for_image", 0);
  }
  switch (pipeline_type)
  {
    case PipelineType::kMono:
      svo_ = factory::makeMono(pnh_);
      break;
    case PipelineType::kStereo:
      svo_ = factory::makeStereo(pnh_);
      break;
    case PipelineType::kArray:
      svo_ = factory::makeArray(pnh_);
      break;
    default:
      LOG(FATAL) << "Unknown pipeline";
      break;
  }
  ncam_ = svo_->getNCamera();

  visualizer_.reset(
        new Visualizer(svo_->options_.trace_dir, pnh_, ncam_->getNumCameras()));
  pub_pampc_landmarks_ =
      pnh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("pampc_landmarks", 2);

  if(vk::param<bool>(pnh_, "use_imu", false))
  {
    imu_handler_ = factory::getImuHandler(pnh_);
    svo_->imu_handler_ = imu_handler_;
  }

  if(vk::param<bool>(pnh_, "use_ceres_backend", false))
  {
    ceres_backend_interface_ = ceres_backend_factory::makeBackend(pnh_,ncam_);
    if(imu_handler_){
      svo_->setBundleAdjuster(ceres_backend_interface_);
      ceres_backend_interface_->setImu(imu_handler_);
      ceres_backend_interface_->makePublisher(pnh_, ceres_backend_publisher_);
    }
    else
    {
      SVO_ERROR_STREAM("Cannot use ceres backend without using imu");
    }
  }
#ifdef SVO_USE_GTSAM_BACKEND
  if(vk::param<bool>(pnh_, "use_backend", false))
  {
    backend_interface_ = svo::backend_factory::makeBackend(pnh_);
    ceres_backend_publisher_.reset(new CeresBackendPublisher(svo_->options_.trace_dir, pnh_));
    svo_->setBundleAdjuster(backend_interface_);
    backend_interface_->imu_handler_ = imu_handler_;
  }
#endif
  if(vk::param<bool>(pnh_, "runlc", false))
  {
#ifdef SVO_LOOP_CLOSING
    LoopClosingPtr loop_closing_ptr =
        factory::getLoopClosingModule(pnh_, svo_->getNCamera());
    svo_->lc_ = std::move(loop_closing_ptr);
    CHECK(svo_->depth_filter_->options_.extra_map_points)
        << "The depth filter seems to be initialized without extra map points.";
#else
    LOG(FATAL) << "You have to enable loop closing in svo_cmake.";
#endif
  }

  if(vk::param<bool>(pnh_, "use_global_map", false))
  {
#ifdef SVO_GLOBAL_MAP
    svo_->global_map_ = factory::getGlobalMap(pnh_, svo_->getNCamera());
    if (imu_handler_)
    {
      svo_->global_map_->initializeIMUParams(imu_handler_->imu_calib_,
                                             imu_handler_->imu_init_);
    }
#else
    LOG(FATAL) << "You have to enable global map in cmake";
#endif
  }

  // A manual initialization workflow must not accumulate an IMU-only backend
  // during takeoff/approach before its explicit reset/start request.
  if (!vk::param<bool>(pnh_, "start_paused", false))
    svo_->start();
}

SvoInterface::~SvoInterface()
{
  if (imu_thread_)
    imu_thread_->join();
  if (image_thread_)
    image_thread_->join();
  VLOG(1) << "Destructed SVO.";
}

void SvoInterface::traceProcessingStage(const char* stage, int64_t timestamp_nanoseconds)
{
  if (!trace_processing_stages_)
    return;
  std_msgs::String message;
  message.data = "frame_ns=" + std::to_string(timestamp_nanoseconds) + " stage=" + stage;
  pub_processing_stage_.publish(message);
}

void SvoInterface::ensureBackend()
{
  if (!svo_->isBackendValid())
  {
    if (vk::param<bool>(pnh_, "use_ceres_backend", false, true))
    {
      ceres_backend_interface_ =
          ceres_backend_factory::makeBackend(pnh_, ncam_);
      if (imu_handler_)
      {
        svo_->setBundleAdjuster(ceres_backend_interface_);
        ceres_backend_interface_->setImu(imu_handler_);
        ceres_backend_interface_->makePublisher(pnh_, ceres_backend_publisher_);
      }
      else
      {
        SVO_ERROR_STREAM("Cannot use ceres backend without using imu");
      }
    }
  }
}

void SvoInterface::processImageBundle(
    const std::vector<cv::Mat>& images,
    const int64_t timestamp_nanoseconds)
{
  const bool first_initialization_image = !svo_->hasStarted();
  ensureBackend();
  svo_->addImageBundle(images, timestamp_nanoseconds);
  if (first_initialization_image && svo_->getLastFrames())
  {
    const auto frames = svo_->getLastFrames();
    const auto pose = frames->get_T_W_B();
    const auto& q = pose.getRotation().toImplementation();
    ROS_INFO_STREAM("SVO_INIT_FRAME phase=first_reference image_ns="
        << timestamp_nanoseconds << " bundle=" << frames->getBundleId()
        << " backend_valid=" << svo_->isBackendValid()
        << " p_wi=[" << pose.getPosition().transpose()
        << "] q_wi_wxyz=[" << q.w() << ' ' << q.x() << ' '
        << q.y() << ' ' << q.z() << ']');
  }
}

void SvoInterface::publishResults(
    const std::vector<cv::Mat>& images,
    const int64_t timestamp_nanoseconds)
{
  CHECK_NOTNULL(svo_.get());
  CHECK_NOTNULL(visualizer_.get());

  visualizer_->img_caption_.clear();
  if (svo_->isBackendValid())
  {
    std::string static_str = ceres_backend_interface_->getStationaryStatusStr();
    visualizer_->img_caption_ = static_str;
  }

  visualizer_->publishSvoInfo(svo_.get(), timestamp_nanoseconds);
  publishPampcLandmarks(timestamp_nanoseconds);
  switch (svo_->stage())
  {
    case Stage::kTracking: {
      Eigen::Matrix<double, 6, 6> covariance;
      covariance.setZero();
      visualizer_->publishImuPose(
            svo_->getLastFrames()->get_T_W_B(), covariance, timestamp_nanoseconds);
      visualizer_->publishCameraPoses(svo_->getLastFrames(), timestamp_nanoseconds);
      visualizer_->visualizeMarkers(
            svo_->getLastFrames(), svo_->closeKeyframes(), svo_->map());
      visualizer_->exportToDense(svo_->getLastFrames());
      bool draw_boundary = false;
      if (svo_->isBackendValid())
      {
        draw_boundary = svo_->getBundleAdjuster()->isFixedToGlobalMap();
      }
      visualizer_->publishImagesWithFeatures(
            svo_->getLastFrames(), timestamp_nanoseconds,
            draw_boundary);
#ifdef SVO_LOOP_CLOSING
      // detections
      if (svo_->lc_)
      {
        visualizer_->publishLoopClosureInfo(
              svo_->lc_->cur_loop_check_viz_info_,
              std::string("loop_query"),
              Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.5);
        visualizer_->publishLoopClosureInfo(
              svo_->lc_->loop_detect_viz_info_, std::string("loop_detection"),
              Eigen::Vector3f(1.0f, 0.0f, 0.0f), 1.0);
        if (svo_->isBackendValid())
        {
          visualizer_->publishLoopClosureInfo(
                svo_->lc_->loop_correction_viz_info_,
                std::string("loop_correction"),
                Eigen::Vector3f(0.0f, 1.0f, 0.0f), 3.0);
        }
        if (svo_->getLastFrames()->at(0)->isKeyframe())
        {
          bool pc_recalculated = visualizer_->publishPoseGraph(
                svo_->lc_->kf_list_,
                svo_->lc_->need_to_update_pose_graph_viz_,
                static_cast<size_t>(svo_->lc_->options_.ignored_past_frames));
          if(pc_recalculated)
          {
            svo_->lc_->need_to_update_pose_graph_viz_ = false;
          }
        }
      }
#endif
#ifdef SVO_GLOBAL_MAP
      if (svo_->global_map_)
      {
        visualizer_->visualizeGlobalMap(*(svo_->global_map_),
                                        std::string("global_vis"),
                                        Eigen::Vector3f(0.0f, 0.0f, 1.0f),
                                        0.3);
        visualizer_->visualizeFixedLandmarks(svo_->getLastFrames()->at(0));
      }
#endif
      break;
    }
    case Stage::kInitializing: {
      visualizer_->publishBundleFeatureTracks(
            svo_->initializer_->frames_ref_, svo_->getLastFrames(),
            timestamp_nanoseconds);
      break;
    }
    case Stage::kPaused:
    case Stage::kRelocalization:
      visualizer_->publishImages(images, timestamp_nanoseconds);
      break;
    default:
      LOG(FATAL) << "Unknown stage";
      break;
  }

#ifdef SVO_USE_GTSAM_BACKEND
  if(svo_->stage() == Stage::kTracking && backend_interface_)
  {
    if(svo_->getLastFrames()->isKeyframe())
    {
      std::lock_guard<std::mutex> estimate_lock(backend_interface_->optimizer_->estimate_mut_);
      const gtsam::Values& state = backend_interface_->optimizer_->estimate_;
      ceres_backend_publisher_->visualizeFrames(state);
      if(backend_interface_->options_.add_imu_factors)
        ceres_backend_publisher_->visualizeVelocity(state);
      ceres_backend_publisher_->visualizePoints(state);
    }
  }
#endif
}

void SvoInterface::publishPampcLandmarks(
    const int64_t timestamp_nanoseconds)
{
  using LandmarkCloud = pcl::PointCloud<pcl::PointXYZ>;
  LandmarkCloud cloud;
  cloud.header.frame_id = Visualizer::kWorldFrame;
  pcl_conversions::toPCL(
      ros::Time().fromNSec(timestamp_nanoseconds), cloud.header.stamp);

  const FrameBundlePtr frames = svo_->getLastFrames();
  if(!frames || frames->size() == 0u)
  {
    pampc_landmark_persistence_.clear();
    pub_pampc_landmarks_.publish(cloud);
    return;
  }

  const FramePtr& frame = frames->at(0);
  std::unordered_set<int> present;
  present.reserve(frame->num_features_);
  for(std::size_t index = 0; index < frame->num_features_; ++index)
  {
    const PointPtr& landmark = frame->landmark_vec_.at(index);
    if(landmark)
      present.insert(landmark->id());
  }

  std::unordered_map<int, int> updated_persistence;
  updated_persistence.reserve(present.size());
  for(const int id : present)
  {
    const auto previous = pampc_landmark_persistence_.find(id);
    updated_persistence[id] =
        previous == pampc_landmark_persistence_.end() ? 1 : previous->second + 1;
  }
  pampc_landmark_persistence_.swap(updated_persistence);

  cloud.reserve(frame->num_features_);
  for(std::size_t index = 0; index < frame->num_features_; ++index)
  {
    const PointPtr& landmark = frame->landmark_vec_.at(index);
    if(!landmark)
      continue;
    const FeatureType type = frame->type_vec_.at(index);
    const bool tracked =
        (!isFixedLandmark(type) && !isMapPoint(type)) ||
        isCornerEdgeletSeed(type);
    if(!tracked || pampc_landmark_persistence_[landmark->id()] < 3)
      continue;

    Eigen::Vector2d reprojected_pixel;
    const auto projection = frame->cam()->project3(
        frame->T_cam_world() * landmark->pos(), &reprojected_pixel);
    if(!projection ||
       !std::isfinite(reprojected_pixel.x()) ||
       !std::isfinite(reprojected_pixel.y()) ||
       (reprojected_pixel - frame->px_vec_.col(index)).norm() > 2.0)
      continue;

    const Eigen::Vector3d& position = landmark->pos();
    if(!position.allFinite())
      continue;
    cloud.push_back(pcl::PointXYZ(
        static_cast<float>(position.x()), static_cast<float>(position.y()),
        static_cast<float>(position.z())));
  }
  pub_pampc_landmarks_.publish(cloud);
}

void SvoInterface::publishStationaryCalibration(
    const std::string& phase, const std::string& detail)
{
  std_msgs::String status;
  status.data = "phase=" + phase + " required=" +
      (require_stationary_calibration_ ? "true" : "false") +
      " duration_s=" + std::to_string(stationary_calibration_duration_) +
      (detail.empty() ? "" : " " + detail);
  pub_stationary_calibration_.publish(status);
  ROS_INFO_STREAM("SVO_STATIONARY_CALIBRATION " << status.data);
}

void SvoInterface::collectStationaryCalibration(const ros::TimerEvent&)
{
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (!stationary_calibration_pending_ || !imu_handler_) return;
  const double now = ros::Time::now().toSec();
  const double end = stationary_calibration_start_ + stationary_calibration_duration_;
  if (now < end) {
    publishStationaryCalibration("collecting", "elapsed_s=" +
        std::to_string(std::max(0.0, now - stationary_calibration_start_)));
    return;
  }
  const auto history = imu_handler_->getMeasurementsCopy();
  ImuMeasurements selected;
  for (const auto& m : history)
    if (m.timestamp_ >= stationary_calibration_start_ && m.timestamp_ <= end &&
        m.linear_acceleration_.allFinite() && m.angular_velocity_.allFinite())
      selected.push_back(m);
  std::sort(selected.begin(), selected.end(), [](const ImuMeasurement& a,
                                                const ImuMeasurement& b) {
    return a.timestamp_ > b.timestamp_;
  });
  double gap = 0.0;
  for (size_t i = 1; i < selected.size(); ++i)
    gap = std::max(gap, selected[i-1].timestamp_ - selected[i].timestamp_);
  const double tolerance = imu_handler_->imu_calib_.max_imu_delta_t;
  const bool covered = selected.size() >= 2u &&
      selected.back().timestamp_ - stationary_calibration_start_ <= tolerance &&
      end - selected.front().timestamp_ <= tolerance && gap <= tolerance;
  if (!covered) {
    if (now > end + 2.0) {
      stationary_calibration_pending_ = false;
      publishStationaryCalibration("failed", "reason=missing_imu_coverage");
    }
    return;
  }
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  for (const auto& m : selected) {
    acc += m.linear_acceleration_;
    gyro += m.angular_velocity_;
  }
  acc /= selected.size(); gyro /= selected.size();
  double avar = 0.0, gvar = 0.0;
  for (const auto& m : selected) {
    avar += (m.linear_acceleration_ - acc).squaredNorm();
    gvar += (m.angular_velocity_ - gyro).squaredNorm();
  }
  const double astd = std::sqrt(avar / selected.size());
  const double gstd = std::sqrt(gvar / selected.size());
  const double gravity = imu_handler_->imu_calib_.gravity_magnitude;
  std::ostringstream stats;
  stats << std::setprecision(17) << "first_imu=" << selected.back().timestamp_
        << " last_imu=" << selected.front().timestamp_ << " count=" << selected.size()
        << " max_gap_s=" << gap << " accel_std=" << astd << " gyro_std=" << gstd
        << " accel_norm=" << acc.norm() << " gyro_mean_norm=" << gyro.norm();
  stationary_calibration_pending_ = false;
  if (astd > vk::param<double>(pnh_, "stationary_calibration_max_accel_std", 0.1) ||
      gstd > vk::param<double>(pnh_, "stationary_calibration_max_gyro_std", 0.015) ||
      gyro.norm() > vk::param<double>(pnh_, "stationary_calibration_max_gyro_mean", 0.03) ||
      std::abs(acc.norm() - gravity) > 0.5) {
    publishStationaryCalibration("failed", "reason=not_stationary_or_implausible " + stats.str());
    return;
  }
  stationary_gyro_bias_ = gyro;
  stationary_accel_bias_ = acc - gravity * acc.normalized();
  stationary_calibration_ready_ = true;
  ceres_backend_interface_->setInitialBiases(stationary_gyro_bias_, stationary_accel_bias_);
  stats << " gyro_bias=[" << stationary_gyro_bias_.transpose()
        << "] accel_bias=[" << stationary_accel_bias_.transpose() << "]";
  publishStationaryCalibration("ready", stats.str());
  // Remain idle until explicit frontend START, even if the drone takes off.
}

bool SvoInterface::collectInitialGravity(const int64_t timestamp_nanoseconds)
{
  if (!gravity_initialization_pending_) return true;
  if (!imu_handler_ || !ceres_backend_interface_) return false;
  const double end = gravity_initialization_start_ + gravity_initialization_duration_;
  const double image_imu_time = timestamp_nanoseconds * 1e-9 -
      imu_handler_->imu_calib_.delay_imu_cam;
  if (image_imu_time < end) return false;
  const auto history = imu_handler_->getMeasurementsCopy();
  if (history.empty() || history.front().timestamp_ < end) return false;
  ImuMeasurements selected;
  for (const auto& m : history)
    if (m.timestamp_ >= gravity_initialization_start_ && m.timestamp_ <= end &&
        m.linear_acceleration_.allFinite() && m.angular_velocity_.allFinite())
      selected.push_back(m);
  std::sort(selected.begin(), selected.end(), [](const ImuMeasurement& a,
                                                const ImuMeasurement& b) {
    return a.timestamp_ > b.timestamp_;
  });
  if (selected.size() < 2u) return false;
  const double tolerance = imu_handler_->imu_calib_.max_imu_delta_t;
  double max_gap = 0.0;
  for (size_t i = 1; i < selected.size(); ++i)
    max_gap = std::max(max_gap, selected[i-1].timestamp_ - selected[i].timestamp_);
  if (selected.back().timestamp_ - gravity_initialization_start_ > tolerance ||
      end - selected.front().timestamp_ > tolerance || max_gap > tolerance)
  {
    ROS_WARN_THROTTLE(1.0, "Fresh gravity window has missing IMU coverage; not using older history");
    return false;
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  double gyro_squared = 0.0;
  for (const auto& m : selected) {
    mean += m.linear_acceleration_;
    gyro_squared += m.angular_velocity_.squaredNorm();
  }
  mean /= selected.size();
  if (mean.norm() < 1.0) return false;
  double accel_variance = 0.0;
  for (const auto& m : selected)
    accel_variance += (m.linear_acceleration_ - mean).squaredNorm();
  gravity_initialization_measurements_ = selected;
  // Opt in only when this window is stationary (e.g. floor bench), not a hover.
  ceres_backend_interface_->setInitialGravityMeasurements(selected,
      !stationary_calibration_ready_ &&
          vk::param<bool>(pnh_, "initialize_gyro_bias_from_gravity_window", false),
      !stationary_calibration_ready_ &&
          vk::param<bool>(pnh_, "initialize_radial_accel_bias_from_gravity_window", false));
  gravity_initialization_pending_ = false;
  std::ostringstream out;
  out << std::setprecision(17)
      << "phase=ready requested_start=" << gravity_initialization_start_
      << " requested_end=" << end
      << " first_imu=" << selected.back().timestamp_
      << " last_imu=" << selected.front().timestamp_
      << " count=" << selected.size() << " max_gap_s=" << max_gap
      << " mean_accel_imu=[" << mean.transpose() << "]"
      << " accel_std_norm=" << std::sqrt(accel_variance / selected.size())
      << " gyro_rms_rad_s=" << std::sqrt(gyro_squared / selected.size());
  std_msgs::String status; status.data = out.str();
  pub_gravity_initialization_.publish(status);
  ROS_INFO_STREAM("SVO_INIT_GRAVITY " << status.data);
  return true;
}

bool SvoInterface::setImuPrior(const int64_t timestamp_nanoseconds)
{
  if (!collectInitialGravity(timestamp_nanoseconds)) return false;
  if(svo_->getBundleAdjuster())
  {
    //if we use backend, this will take care of setting priors
    if(!svo_->hasStarted())
    {
      //when starting up, make sure we already have IMU measurements
      if(imu_handler_->getMeasurementsCopy().size() < 10u)
      {
        return false;
      }
    }
    return true;
  }

  if(imu_handler_ && !svo_->hasStarted() && set_initial_attitude_from_gravity_)
  {
    // set initial orientation
    Quaternion R_imu_world;
    if(imu_handler_->getInitialAttitude(
         timestamp_nanoseconds * common::conversions::kNanoSecondsToSeconds,
         R_imu_world))
    {
      VLOG(3) << "Set initial orientation from accelerometer measurements.";
      svo_->setRotationPrior(R_imu_world);
    }
    else
    {
      return false;
    }
  }
  else if(imu_handler_ && svo_->getLastFrames())
  {
    // set incremental rotation prior
    Quaternion R_lastimu_newimu;
    if(imu_handler_->getRelativeRotationPrior(
         svo_->getLastFrames()->getMinTimestampNanoseconds() *
         common::conversions::kNanoSecondsToSeconds,
         timestamp_nanoseconds * common::conversions::kNanoSecondsToSeconds,
         false, R_lastimu_newimu))
    {
      VLOG(3) << "Set incremental rotation prior from IMU.";
      svo_->setRotationIncrementPrior(R_lastimu_newimu);
    }
  }
  return true;
}

namespace {
// TX2_MONO_PROFILE_V1
// Image transport's JPEG decoding happens BEFORE this callback. Entry age
// includes render/network/queue/decode; do not mislabel it as queue time alone.
class MonoCallbackTiming {
 public:
  explicit MonoCallbackTiming(const ros::Time& stamp)
      : start_(Clock::now()), last_(start_) {
    static ros::NodeHandle private_nh("~");
    static ros::Publisher publisher = private_nh.advertise<std_msgs::Float64MultiArray>("mono_timing", 10);
    publisher_ = publisher;
    message_.data.assign(14, 0.0);
    message_.data[0] = stamp.toSec();
    message_.data[1] = ros::Time::now().toSec();
    message_.data[2] = (message_.data[1]-message_.data[0])*1000.;
  }
  void mark(size_t index) {
    const auto now = Clock::now();
    message_.data[index] = std::chrono::duration<double, std::milli>(now-last_).count();
    last_ = now;
  }
  void complete(int stage) { message_.data[12]=stage; message_.data[13]=1.; }
  ~MonoCallbackTiming() {
    message_.data[10] = std::chrono::duration<double, std::milli>(Clock::now()-start_).count();
    message_.data[11] = ros::Time::now().toSec();
    publisher_.publish(message_);
  }
 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_, last_;
  ros::Publisher publisher_;
  std_msgs::Float64MultiArray message_;
};

void processMonoImage(
    svo::SvoInterface* interface,
    const sensor_msgs::ImageConstPtr& msg,
    const sensor_msgs::ImageConstPtr& depth_msg)
{
  MonoCallbackTiming timing(msg->header.stamp);
  const int64_t frame_stamp = msg->header.stamp.toNSec();
  interface->traceProcessingStage("waiting_for_pipeline_lock", frame_stamp);
  std::lock_guard<std::mutex> pipeline_lock(interface->pipeline_mutex_);
  timing.mark(3);
  if(interface->idle_)
  {
    interface->traceProcessingStage("idle", frame_stamp);
    return;
  }

  interface->traceProcessingStage("copy_image", frame_stamp);
  cv::Mat image;
  try
  {
    image = cv_bridge::toCvCopy(msg)->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    interface->traceProcessingStage("image_conversion_failed", frame_stamp);
    return;
  }

  if(depth_msg)
  {
    interface->traceProcessingStage("copy_depth", frame_stamp);
    try
    {
      const cv::Mat depth = cv_bridge::toCvCopy(depth_msg, "32FC1")->image;
      if(depth.size() != image.size())
      {
        ROS_ERROR("SVO OneShot depth dimensions (%dx%d) do not match RGB (%dx%d)",
                  depth.cols, depth.rows, image.cols, image.rows);
        interface->traceProcessingStage("depth_dimensions_failed", frame_stamp);
        return;
      }
      interface->svo_->setDepthPriorImage(depth);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("SVO OneShot depth cv_bridge exception: %s", e.what());
      interface->traceProcessingStage("depth_conversion_failed", frame_stamp);
      return;
    }
  }

  std::vector<cv::Mat> images;
  images.push_back(image.clone());
  timing.mark(4);

  interface->traceProcessingStage("imu_prior", frame_stamp);
  if(!interface->setImuPrior(msg->header.stamp.toNSec()))
  {
    VLOG(3) << "Could not align gravity! Attempting again in next iteration.";
    interface->traceProcessingStage("waiting_for_imu_prior", frame_stamp);
    return;
  }

  timing.mark(5);
  interface->traceProcessingStage("preprocess", frame_stamp);
  interface->imageCallbackPreprocessing(msg->header.stamp.toNSec());
  timing.mark(6);
  interface->traceProcessingStage("process_image_bundle", frame_stamp);
  interface->processImageBundle(images, msg->header.stamp.toNSec());
  timing.mark(7);
  interface->traceProcessingStage("publish_results", frame_stamp);
  interface->publishResults(images, msg->header.stamp.toNSec());
  timing.mark(8);

  if(interface->svo_->stage() == Stage::kPaused &&
     interface->automatic_reinitialization_)
    interface->svo_->start();

  interface->traceProcessingStage("postprocess", frame_stamp);
  interface->imageCallbackPostprocessing();
  timing.mark(9);
  timing.complete(static_cast<int>(interface->svo_->stage()));
  interface->traceProcessingStage("callback_finished", frame_stamp);
}
}  // namespace

void SvoInterface::monoCallback(const sensor_msgs::ImageConstPtr& msg)
{
  processMonoImage(this, msg, sensor_msgs::ImageConstPtr());
}

void SvoInterface::monoDepthCallback(
    const sensor_msgs::ImageConstPtr& image_msg,
    const sensor_msgs::ImageConstPtr& depth_msg)
{
  processMonoImage(this, image_msg, depth_msg);
}

void SvoInterface::stereoCallback(
    const sensor_msgs::ImageConstPtr& msg0,
    const sensor_msgs::ImageConstPtr& msg1)
{
  std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex_);
  if(idle_)
    return;

  cv::Mat img0, img1;
  try {
    img0 = cv_bridge::toCvShare(msg0, "mono8")->image;
    img1 = cv_bridge::toCvShare(msg1, "mono8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  if(!setImuPrior(msg0->header.stamp.toNSec()))
  {
    VLOG(3) << "Could not align gravity! Attempting again in next iteration.";
    return;
  }

  imageCallbackPreprocessing(msg0->header.stamp.toNSec());

  processImageBundle({img0, img1}, msg0->header.stamp.toNSec());
  publishResults({img0, img1}, msg0->header.stamp.toNSec());

  if(svo_->stage() == Stage::kPaused && automatic_reinitialization_)
    svo_->start();

  imageCallbackPostprocessing();
}

void SvoInterface::imuCallback(const sensor_msgs::ImuConstPtr& msg)
{
  const Eigen::Vector3d omega_imu(
        msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  const Eigen::Vector3d lin_acc_imu(
        msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const ImuMeasurement m(msg->header.stamp.toSec(), omega_imu, lin_acc_imu);
  if(imu_handler_)
    imu_handler_->addImuMeasurement(m);
  else
    SVO_ERROR_STREAM("SvoNode has no ImuHandler");
}

void SvoInterface::inputKeyCallback(const std_msgs::StringConstPtr& key_input)
{
  if (key_input->data.empty())
    return;
  const auto received = std::chrono::steady_clock::now();
  std_msgs::String status;
  status.data = "key=" + key_input->data + " phase=received_waiting_for_pipeline_lock";
  pub_remote_key_status_.publish(status);
  ROS_INFO_STREAM("SVO remote key: " << status.data);
  // Receipt is diagnostic only. Keep the real acknowledgement after handling
  // the command under the image-processing lock.
  std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex_);
  const double lock_wait_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - received).count();
  status.data = "key=" + key_input->data + " phase=lock_acquired wait_ms=" + std::to_string(lock_wait_ms);
  pub_remote_key_status_.publish(status);
  ROS_INFO_STREAM("SVO remote key: " << status.data);
  std::string remote_input = key_input->data;
  char input = remote_input.c_str()[0];
  switch(input)
  {
    case 'q':
      quit_ = true;
      SVO_INFO_STREAM("SVO user input: QUIT");
      break;
    case 'b':
      if (!imu_handler_ || !vk::param<bool>(pnh_, "use_ceres_backend", false)) {
        publishStationaryCalibration("failed", "reason=requires_imu_and_ceres");
        return;
      }
      svo_->resetForInitialization();
      ceres_backend_interface_.reset();
      ensureBackend();
      gravity_initialization_pending_ = false;
      gravity_initialization_measurements_.clear();
      stationary_calibration_ready_ = false;
      stationary_calibration_pending_ = true;
      stationary_calibration_start_ = ros::Time::now().toSec();
      idle_ = true;  // IMU collection only: do not create a visual map or states.
      pampc_landmark_persistence_.clear();
      publishStationaryCalibration("collecting", "elapsed_s=0");
      break;
    case 'v':
      // Explicit bench opt-in; operator confirms disarmed and stationary.
      if (!vk::param<bool>(pnh_, "stationary_box_test_enabled", false) ||
          !ceres_backend_interface_ || svo_->stage() != Stage::kTracking) {
        ROS_WARN("SVO_BOX_TEST rejected: require bench opt-in and tracking backend");
        return;
      }
      if (!ceres_backend_interface_->requestStationaryBoxTest(
              ros::Time::now().toSec(), 10.0)) {
        ROS_WARN("SVO_BOX_TEST rejected: backend not ready");
        return;
      }
      ROS_WARN("SVO_BOX_TEST started: keep disarmed and stationary for 10 seconds");
      break;
    case 'V':
      if (ceres_backend_interface_)
        ceres_backend_interface_->requestStationaryBoxTest(0, 0);
      break;
    case 'r':
      if (stationary_calibration_pending_) {
        stationary_calibration_pending_ = false;
        publishStationaryCalibration("cancelled", "reason=reset_during_collection");
      }
      gravity_initialization_pending_ = false;
      gravity_initialization_measurements_.clear();
      svo_->reset();
      pampc_landmark_persistence_.clear();
      // FrameHandlerBase::reset() is deferred until the end of the next
      // frame. Keep callbacks enabled so that frame can retire the frontend
      // map and Ceres backend before a subsequent START command.
      idle_ = false;
      SVO_INFO_STREAM("SVO user input: RESET");
      break;
    case 's':
      if (stationary_calibration_pending_ ||
          (require_stationary_calibration_ && !stationary_calibration_ready_)) {
        ROS_WARN("SVO START rejected: complete stationary backend calibration first");
        return;
      }
      // Retire the old map/backend before the next image selects its reference.
      // processImageBundle will create the fresh backend BEFORE addFrameBundle,
      // allowing its gravity prior to set the very first reference pose.
      svo_->resetForInitialization();
      ceres_backend_interface_.reset();
      // Construct the backend before beginning the timed hover window. Its
      // setup cost must not separate the first two visual reference images.
      ensureBackend();
      if (stationary_calibration_ready_) {
        ceres_backend_interface_->setInitialBiases(
            stationary_gyro_bias_, stationary_accel_bias_);
        publishStationaryCalibration("ready", "reused_for_visual_initialization=true");
      }
      gravity_initialization_measurements_.clear();
      gravity_initialization_pending_ = gravity_initialization_duration_ > 0.0;
      gravity_initialization_start_ = ros::Time::now().toSec();
      if (gravity_initialization_pending_)
      {
        if (!imu_handler_ || !ceres_backend_interface_) {
          ROS_ERROR("Fresh gravity initialization requires IMU and Ceres backend");
          return;
        }
        std_msgs::String status;
        status.data = "phase=collecting start=" + std::to_string(gravity_initialization_start_) +
            " duration=" + std::to_string(gravity_initialization_duration_);
        pub_gravity_initialization_.publish(status);
        ROS_INFO_STREAM("SVO_INIT_GRAVITY " << status.data);
      }
      svo_->start();
      pampc_landmark_persistence_.clear();
      idle_ = false;
      SVO_INFO_STREAM("SVO user input: START (old frontend/backend cleared before first reference)");
      break;
     case 'c':
      svo_->setCompensation(true);
      SVO_INFO_STREAM("Enabled affine compensation.");
      break;
     case 'C':
      svo_->setCompensation(false);
      SVO_INFO_STREAM("Disabled affine compensation.");
      break;
    default: ;
  }
  if (input == 'r' || input == 's' || input == 'b')
  {
    // Confirm that this SVO process handled the command. A ROS publisher's
    // connection count can otherwise be satisfied by rosbag alone.
    std_msgs::UInt32 ack;
    ack.data = ++remote_key_ack_count_;
    pub_remote_key_ack_.publish(ack);
    status.data = "key=" + remote_input + " phase=handled ack=" + std::to_string(ack.data);
    pub_remote_key_status_.publish(status);
    ROS_INFO_STREAM("SVO remote key: " << status.data);
  }
}

void SvoInterface::subscribeImu()
{
  imu_thread_ = std::unique_ptr<std::thread>(
        new std::thread(&SvoInterface::imuLoop, this));
  sleep(3);
}

void SvoInterface::subscribeImage()
{
  if(pipeline_type_ == PipelineType::kMono)
    image_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&SvoInterface::monoLoop, this));
  else if(pipeline_type_ == PipelineType::kStereo)
    image_thread_ = std::unique_ptr<std::thread>(
        new std::thread(&SvoInterface::stereoLoop, this));
}

void SvoInterface::subscribeRemoteKey()
{
  std::string remote_key_topic =
      vk::param<std::string>(pnh_, "remote_key_topic", "svo/remote_key");
  std::string remote_key_ack_topic =
      vk::param<std::string>(pnh_, "remote_key_ack_topic", "/svo/remote_key_ack");
  pub_remote_key_ack_ = nh_.advertise<std_msgs::UInt32>(remote_key_ack_topic, 10, false);
  sub_remote_key_ =
      nh_.subscribe(remote_key_topic, 5, &svo::SvoInterface::inputKeyCallback, this);
}

void SvoInterface::imuLoop()
{
  SVO_INFO_STREAM("SvoNode: Started IMU loop.");
  ros::NodeHandle nh;
  ros::CallbackQueue queue;
  nh.setCallbackQueue(&queue);
  std::string imu_topic = vk::param<std::string>(pnh_, "imu_topic", "imu");
  ros::Subscriber sub_imu =
      nh.subscribe(imu_topic, 10, &svo::SvoInterface::imuCallback, this);
  while(ros::ok() && !quit_)
  {
    queue.callAvailable(ros::WallDuration(0.1));
  }
}

void SvoInterface::monoLoop()
{
  SVO_INFO_STREAM("SvoNode: Started Image loop.");

  ros::NodeHandle nh;
  ros::CallbackQueue queue;
  nh.setCallbackQueue(&queue);

  image_transport::ImageTransport it(nh);
  const std::string image_topic =
      vk::param<std::string>(pnh_, "cam0_topic", "camera/image_raw");
  const std::string depth_topic =
      vk::param<std::string>(pnh_, "depth_topic", "");
  image_transport::Subscriber it_sub;
  const int initialization_queue_size = vk::param<int>(pnh_, "image_queue_size", 5);
  const int tracking_queue_size = vk::param<int>(
      pnh_, "image_tracking_queue_size", initialization_queue_size);
  int active_queue_size = initialization_queue_size;
  ros::Publisher queue_size_pub = pnh_.advertise<std_msgs::UInt32>(
      "image_queue_size_active", 1, true);
  std::unique_ptr<image_transport::SubscriberFilter> image_filter;
  std::unique_ptr<image_transport::SubscriberFilter> depth_filter;
  typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
  std::unique_ptr<message_filters::Synchronizer<ExactPolicy>> depth_sync;
  if(depth_topic.empty())
  {
    // Retain a small burst while initialization blocks the image callback.
    if (initialization_queue_size < 1 || tracking_queue_size < 1) {
      ROS_FATAL("Image queue sizes must be positive");
      return;
    }
    ROS_INFO("Mono image subscriber queue: initialization=%d tracking=%d",
             initialization_queue_size, tracking_queue_size);
    it_sub = it.subscribe(image_topic, active_queue_size, &svo::SvoInterface::monoCallback, this);
    std_msgs::UInt32 message;
    message.data = active_queue_size;
    queue_size_pub.publish(message);
  }
  else
  {
    image_filter.reset(new image_transport::SubscriberFilter(it, image_topic, 5));
    depth_filter.reset(new image_transport::SubscriberFilter(it, depth_topic, 5));
    depth_sync.reset(new message_filters::Synchronizer<ExactPolicy>(
        ExactPolicy(10), *image_filter, *depth_filter));
    depth_sync->registerCallback(
        boost::bind(&svo::SvoInterface::monoDepthCallback, this, _1, _2));
    SVO_INFO_STREAM("SvoNode: synchronized OneShot depth topic " << depth_topic);
  }

  while(ros::ok() && !quit_)
  {
    // Check the stage after each image, rather than draining a captured batch
    // of initialization callbacks before switching to the tracking queue.
    if (depth_topic.empty())
      queue.callOne(ros::WallDuration(0.1));
    else
      queue.callAvailable(ros::WallDuration(0.1));
    if (depth_topic.empty()) {
      int desired_queue_size;
      {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        desired_queue_size = (!idle_ && svo_->stage() == Stage::kTracking)
            ? tracking_queue_size : initialization_queue_size;
      }
      if (desired_queue_size != active_queue_size) {
        it_sub.shutdown();
        // This callback queue belongs only to the image loop. IMU callbacks
        // and the IMU history remain untouched, as do map/backend state.
        queue.clear();
        active_queue_size = desired_queue_size;
        it_sub = it.subscribe(image_topic, active_queue_size,
                              &svo::SvoInterface::monoCallback, this);
        std_msgs::UInt32 message;
        message.data = active_queue_size;
        queue_size_pub.publish(message);
        ROS_INFO("Mono image subscriber queue switched to %d", active_queue_size);
      }
    }
  }
}

void SvoInterface::stereoLoop()
{
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;

  ros::NodeHandle nh(nh_, "image_thread");
  ros::CallbackQueue queue;
  nh.setCallbackQueue(&queue);

  // subscribe to cam msgs
  std::string cam0_topic(vk::param<std::string>(pnh_, "cam0_topic", "/cam0/image_raw"));
  std::string cam1_topic(vk::param<std::string>(pnh_, "cam1_topic", "/cam1/image_raw"));
  image_transport::ImageTransport it(nh);
  image_transport::SubscriberFilter sub0(it, cam0_topic, 1, std::string("raw"));
  image_transport::SubscriberFilter sub1(it, cam1_topic, 1, std::string("raw"));
  ExactSync sync_sub(ExactPolicy(5), sub0, sub1);
  sync_sub.registerCallback(boost::bind(&svo::SvoInterface::stereoCallback, this, _1, _2));

  while(ros::ok() && !quit_)
  {
    queue.callAvailable(ros::WallDuration(0.1));
  }
}

} // namespace svo
