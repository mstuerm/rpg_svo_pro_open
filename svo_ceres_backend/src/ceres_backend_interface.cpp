#include "svo/ceres_backend_interface.hpp"

#include <svo/common/conversions.h>
#include <svo/common/frame.h>
#include <svo/map.h>
#include <svo/imu_handler.h>
#include <svo/global.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <set>
#include <sstream>
#include <sys/statvfs.h>

#include "svo/motion_detector.hpp"
#include "svo/outlier_rejection.hpp"

//! @todo Esimation of extrinsics not tested!
DEFINE_double(extrinsics_sigma_rel_translation, 0.0,
              "Relative translation sigma (temporal) of camera extrinsics");
DEFINE_double(extrinsics_sigma_rel_orientation, 0.0,
              "Relative translation sigma (temporal) of camera extrinsics");

namespace svo
{
CeresBackendInterface::CeresBackendInterface(
    const CeresBackendInterfaceOptions& options,
    const CeresBackendOptions& optimizer_options,
    const MotionDetectorOptions& motion_detector_options,
    const CameraBundlePtr& camera_bundle)
  : options_(options), optimizer_options_(optimizer_options)
{
  type_ = BundleAdjustmentType::kCeres;

  // Setup modules
  if (options_.use_zero_motion_detection)
  {
    motion_detector_.reset(new MotionDetector(motion_detector_options));
  }
  if (options_.use_outlier_rejection)
  {
    outlier_rejection_.reset(
        new OutlierRejection(options_.outlier_rejection_px_threshold));
  }
  // Cameras -------------------------------------------------------------------
  // For now do not estimate extrinsics. (NOT TESTED!)
  if (options.refine_extrinsics)
  {
    ExtrinsicsEstimationParametersVec extrinsics_estimation_parameters(
        camera_bundle->getNumCameras(),
        ExtrinsicsEstimationParameters(options.extrinsics_pos_sigma_meter,
                                       options.extrinsics_rot_sigma_rad,
                                       FLAGS_extrinsics_sigma_rel_translation,
                                       FLAGS_extrinsics_sigma_rel_orientation));
    backend_.addCameraBundle(extrinsics_estimation_parameters, camera_bundle);
  }
  else
  {
    ExtrinsicsEstimationParametersVec extrinsics_estimation_parameters(
        camera_bundle->getNumCameras(),
        ExtrinsicsEstimationParameters(0.0, 0.0, 0.0, 0.0));
    backend_.addCameraBundle(extrinsics_estimation_parameters, camera_bundle);
  }

  // Soft time limit for backend
  if (optimizer_options_.max_iteration_time > 0.0)
  {
    backend_.setOptimizationTimeLimit(optimizer_options_.max_iteration_time, 1);
  }

  backend_.min_num_3d_points_for_fixation_ =
      optimizer_options_.remove_fixation_min_num_fixed_landmarks_;
}

bool CeresBackendInterface::requestStationaryBoxTest(
    double image_start_stamp, double duration_seconds)
{
  std::lock_guard<std::mutex> lock(mutex_backend_);
  if (duration_seconds == 0) {
    stationary_box_test_active_ = false;
    LOG(INFO) << "SVO_BOX_TEST phase=cancelled";
    return true;
  }
  if (!std::isfinite(image_start_stamp) || !std::isfinite(duration_seconds) ||
      duration_seconds <= 0 || duration_seconds > 10 || backend_.numFrames() == 0)
    return false;
  stationary_box_test_image_start_ = image_start_stamp;
  stationary_box_test_duration_ = duration_seconds;
  stationary_box_test_deadline_ = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(static_cast<int>(duration_seconds * 1000));
  stationary_box_test_active_ = true;
  LOG(INFO) << "SVO_BOX_TEST phase=armed duration_s=" << duration_seconds
            << " sigma_m_s=0.01 image_start=" << std::setprecision(17) << image_start_stamp;
  return true;
}

CeresBackendInterface::~CeresBackendInterface()
{
  if (thread_ != nullptr)
  {
    quitThread();
  }
}

// Called under mutex_backend_. Inspect state only; do not repair or filter it.
void CeresBackendInterface::traceHandoff(
    const char* phase, const FrameBundlePtr& current,
    const FrameBundlePtr& previous, const Map::Ptr& map) const
{
  std::ostringstream out;
  out << std::setprecision(9) << "SVO_HANDOFF phase=" << phase
      << " image_ns=" << (current ? current->getMinTimestampNanoseconds() : 0)
      << " bundle=" << (current ? current->getBundleId() : -1)
      << " imu_bundle=" << last_added_nframe_imu_
      << " visual_bundle=" << last_added_nframe_images_
      << " optimized_bundle=" << last_optimized_nframe_.load()
      << " applied_bundle=" << last_updated_nframe_
      << " backend_frames=" << backend_.numFrames()
      << " active_kfs=[";
  for (const auto& frame : active_keyframes_)
    if (frame) out << frame->id() << ':' << frame->bundleId() << ',';
  out << ']';
  const auto describe_frame = [&out](const char* role, const FramePtr& frame) {
    if (!frame) { out << ' ' << role << "=null"; return; }
    const auto pose = frame->T_world_imu();
    const auto& q = pose.getRotation().toImplementation();
    out << ' ' << role << "={id:" << frame->id()
        << ",bundle:" << frame->bundleId() << ",kf:" << frame->isKeyframe()
        << ",features:" << frame->numFeatures()
        << ",landmarks:" << frame->numLandmarks()
        << ",in_ba:" << frame->numLandmarksInBA()
        << ",p_wi:[" << pose.getPosition().transpose()
        << "],q_wi_wxyz:[" << q.w() << ' ' << q.x() << ' ' << q.y()
        << ' ' << q.z() << "]}";
  };
  if (current)
    for (const auto& frame : *current) describe_frame("current", frame);
  if (previous)
    for (const auto& frame : *previous) describe_frame("previous", frame);
  std::set<int> points;
  size_t finite = 0, visible = 0, samples = 0;
  if (map)
  {
    out << " map_kfs=" << map->size();
    for (const auto& entry : map->keyframes_)
    {
      describe_frame("map_kf", entry.second);
      if (!entry.second) continue;
      for (const auto& point : entry.second->landmark_vec_)
      {
        if (!point || !points.insert(point->id()).second) continue;
        if (point->pos_.allFinite())
        {
          ++finite;
          if (current && !current->empty() && current->at(0)->isVisible(point->pos_))
            ++visible;
        }
        if (samples++ < 3)
          out << " point={id:" << point->id() << ",p_w:["
              << point->pos_.transpose() << "]}";
      }
    }
    out << " unique_map_landmarks=" << points.size()
        << " finite_map_landmarks=" << finite
        << " geometrically_visible_in_current=" << visible;
  }
  LOG(INFO) << out.str();
}

std::string CeresBackendInterface::frontendConsistencySnapshot(
    const FrameBundlePtr& previous, const Map::Ptr& map) const
{
  std::map<int, FramePtr> frames;
  if (map) for (const auto& entry : map->keyframes_)
    if (entry.second) frames.emplace(entry.second->id(), entry.second);
  if (previous) for (const auto& frame : *previous)
    if (frame) frames.emplace(frame->id(), frame);
  const auto pose_json = [](const Transformation& pose) {
    std::ostringstream stream;
    const auto p = pose.getPosition();
    const auto q = pose.getRotation().toImplementation();
    if (!p.allFinite() || !q.coeffs().allFinite()) return std::string("null");
    stream << std::setprecision(12) << "{\"p\":[" << p.x() << ',' << p.y() << ',' << p.z()
           << "],\"q_xyzw\":[" << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << "]}";
    return stream.str();
  };
  std::ostringstream out;
  out << std::setprecision(12) << "{\"applied_bundle\":" << last_updated_nframe_
      << ",\"optimized_bundle\":" << last_optimized_nframe_.load() << ",\"frames\":[";
  bool first = true;
  size_t count = 0;
  std::map<int, PointPtr> points;
  for (const auto& item : frames) {
    const auto& frame = item.second;
    if (count++ >= 32) break;
    if (!first) { out << ','; } first = false;
    out << "{\"frame_id\":" << frame->id() << ",\"bundle\":" << frame->bundleId()
        << ",\"backend_id\":\"" << createNFrameId(frame->bundleId()).asInteger()
        << "\",\"keyframe\":" << (frame->isKeyframe() ? "true" : "false")
        << ",\"features\":" << frame->numFeatures() << ",\"landmarks\":" << frame->numLandmarks()
        << ",\"landmarks_in_ba\":" << frame->numLandmarksInBA()
        << ",\"frontend_pose\":" << pose_json(frame->T_world_imu());
    Transformation backend_pose;
    out << ",\"backend_pose\":";
    if (backend_.get_T_WS(createNFrameId(frame->bundleId()), backend_pose)) out << pose_json(backend_pose);
    else out << "null";
    out << '}';
    for (const auto& point : frame->landmark_vec_)
      if (point) points.emplace(point->id(), point);
  }
  out << "],\"landmarks\":[";
  first = true; count = 0;
  for (const auto& item : points) {
    if (count++ >= 24) break;
    if (!first) { out << ','; } first = false;
    out << "{\"point_id\":" << item.first << ",\"p\":";
    const auto p = item.second->pos_;
    if (p.allFinite()) out << '[' << p.x() << ',' << p.y() << ',' << p.z() << ']';
    else out << "null";
    out << '}';
  }
  out << "]}";
  return out.str();
}

// Get a motion prior for new_frames and update the frontend map and last_frames
// (note that map is not used, but actually all keyframes in map are updated
//  in call to updateActiveKeyframes() )
void CeresBackendInterface::loadMapFromBundleAdjustment(
    const FrameBundlePtr& new_frames, const FrameBundlePtr& last_frames,
    const Map::Ptr& map, bool& have_motion_prior)
{
  if (stop_thread_)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_backend_);

  const auto feedback_now = std::chrono::steady_clock::now();
  const bool feedback_sample = options_.trace_consistency && map && map->size() > 0 &&
      feedback_now - consistency_last_feedback_ >= std::chrono::seconds(1);
  const std::string feedback_before = feedback_sample
      ? frontendConsistencySnapshot(last_frames, map) : std::string();

  const bool trace = options_.trace_handoff && map && map->size() > 0 &&
      handoff_trace_frames_ < options_.trace_handoff_frames;
  if (trace)
  {
    ++handoff_trace_frames_;
    traceHandoff("before_imu_prior", new_frames, last_frames, map);
  }

  // Setup motion detector ----------------------------------------------------
  if (motion_detector_)
  {
    motion_detector_->setFrames(last_frames, new_frames);
  }

  // Adding new state to backend ---------------------------------------------
  if (addStatesAndInertialMeasurementsToBackend(new_frames))
  {
    last_added_nframe_imu_ = new_frames->getBundleId();

    // Obtain motion prior ---------------------------------------------------
    updateBundleStateWithBackend(new_frames, true);
    have_motion_prior = true;
  }
  else
  {
    LOG(ERROR) << "Could not add frame bundle " << new_frames->getBundleId()
               << " to backend";
    have_motion_prior = false;
  }

  if (imu_handler_ && imu_handler_->options_.temporal_stationary_check)
  {
    IMUTemporalStatus imu_status =
        imu_handler_->checkTemporalStatus(new_frames->at(0)->getTimestampSec());
    imu_motion_detector_stationary_ =
        (imu_status == IMUTemporalStatus::kStationary);
  }

  if (trace) traceHandoff("after_imu_prior", new_frames, last_frames, map);

  // Update Frames and Map ---------------------------------------------------
  if (last_updated_nframe_ == last_optimized_nframe_.load())
  {
    if (trace) traceHandoff("no_optimized_update", new_frames, last_frames, map);
    VLOG(3) << "VIN: No map update available.";
    return;
  }

  // Update SVO Map ----------------------------------------------------------
  {
    Transformation T_WS;
    // Statistics
    int n_frames_updated = 0;

    VLOG(3) << "Updating states with latest results from ceres optimizer.";
    //! @todo this is not very efficient for multiple cameras,
    //! because we update each frame separately
    //! this we we need to get T_WS twice
    //! @todo store framebundles in map to solve problem
    for (FramePtr& keyframe : active_keyframes_)
    {
      DEBUG_CHECK(keyframe) << "Found nullptr keyframe";
      updateFrameStateWithBackend(keyframe, false);
      n_frames_updated++;
    }
    if (trace) traceHandoff("after_keyframe_poses", new_frames, last_frames, map);
    VLOG(3) << "Updated " << n_frames_updated << " frames in map.";

    // Update the 3d points in map of the updated keyframes ----------------
    // Statistics
    backend_.updateAllActivePoints();
    if (trace) traceHandoff("after_landmark_positions", new_frames, last_frames, map);
  }

  // Update last frame bundle ------------------------------------------------
  {
    // Last frames might not be keyframes (are not yet updated => update pose)
    for (FramePtr& last_frame : *last_frames)
    {
      if (!last_frame->isKeyframe())
      {
        updateFrameStateWithBackend(last_frame, true);
      }
    }

    if (trace) traceHandoff("before_outlier_rejection", new_frames, last_frames, map);

    // Remove outliers of last_frames ----------------------------------------
    if (outlier_rejection_)
    {
      if (last_frames)
      {
        size_t n_deleted_edges = 0;
        size_t n_deleted_corners = 0;
        std::vector<int> deleted_points;
        for (FramePtr& frame : *last_frames)
        {
          outlier_rejection_->removeOutliers(*frame, n_deleted_edges,
                                             n_deleted_corners, deleted_points,
                                             !lock_to_fixed_landmarks_);
        }
        //! @todo should we only remove observation but leave points?
        backend_.removePointsByPointIds(deleted_points);
        if (trace)
          LOG(INFO) << "SVO_HANDOFF phase=outlier_rejection image_ns="
                    << new_frames->getMinTimestampNanoseconds()
                    << " bundle=" << new_frames->getBundleId()
                    << " removed_edges=" << n_deleted_edges
                    << " removed_corners=" << n_deleted_corners
                    << " removed_point_ids=" << deleted_points.size();
        VLOG(6) << "Outlier rejection: removed " << n_deleted_edges
                << " edgelets and " << n_deleted_corners << " corners.";
      }
    }

    if (trace) traceHandoff("after_outlier_rejection", new_frames, last_frames, map);

    // The following is not used for the algorithm to work, but updated for
    // completeness.
    SpeedAndBias speed_and_bias;
    bool success =
        backend_.getSpeedAndBias(last_frames->getBundleId(), speed_and_bias);
    DEBUG_CHECK(success) << "Could not get speed and bias estimate from ceres "
                            "optimizer";
    imu_handler_->setAccelerometerBias(speed_and_bias.tail<3>());
    imu_handler_->setGyroscopeBias(speed_and_bias.segment<3>(3));

    // A publisher is optional. In the direct pybind11 path no ROS publisher
    // is constructed, while the optimizer and frontend map handoff are still
    // fully functional.
    if (publisher_)
    {
      publisher_->addFrame(last_added_nframe_imu_);
    }
  }

  // shift state
  last_updated_nframe_ = last_optimized_nframe_.load();
  if (feedback_sample) {
    std::ostringstream record;
    record << "{\"schema\":1,\"kind\":\"frontend_feedback\",\"instance\":\""
           << consistency_instance_ << "\",\"image_stamp_ns\":\""
           << new_frames->getMinTimestampNanoseconds() << "\",\"before\":"
           << feedback_before << ",\"after\":" << frontendConsistencySnapshot(last_frames, map) << '}';
    writeConsistencyRecord(record.str());
    consistency_last_feedback_ = feedback_now;
  }
}

// Add feature correspondences and landmarks to backend
void CeresBackendInterface::bundleAdjustment(const FrameBundlePtr& frame_bundle)
{
  if (stop_thread_)
  {
    return;
  }

  // check for case when IMU measurements could not be added.
  if (last_added_nframe_imu_ == last_added_nframe_images_)
  {
    return;
  }

  /** Uncomment this for testing loop closure
   last_frame_ = frame_bundle->at(0);
   */

  std::lock_guard<std::mutex> lock(mutex_backend_);

  if (options_.trace_handoff && frame_bundle->isKeyframe() &&
      handoff_trace_frames_ < options_.trace_handoff_frames)
    traceHandoff("before_visual_enqueue", frame_bundle, FrameBundlePtr(), Map::Ptr());

  vk::Timer timer;
  timer.start();
  // Checking for zero motion ------------------------------------------------
  bool velocity_prior_added = false;
  if (stationary_box_test_active_ &&
      std::chrono::steady_clock::now() >= stationary_box_test_deadline_) {
    stationary_box_test_active_ = false;
    LOG(INFO) << "SVO_BOX_TEST phase=expired";
  }
  const double image_stamp = frame_bundle->at(0)->getTimestampSec();
  const bool box_test_prior = stationary_box_test_active_ &&
      image_stamp >= stationary_box_test_image_start_ &&
      image_stamp < stationary_box_test_image_start_ + stationary_box_test_duration_;
  if (box_test_prior) {
    velocity_prior_added = backend_.addVelocityPrior(
        createNFrameId(frame_bundle->getBundleId()),
        Eigen::Matrix<FloatType, 3, 1>::Zero(), 0.01);
    if (!velocity_prior_added)
      LOG(ERROR) << "SVO_BOX_TEST phase=prior_failed image=" << image_stamp;
  }
  if (motion_detector_)
  {
    double sigma = 0;
    if (!motion_detector_->isImageMoving(sigma))
    {
      ++no_motion_counter_;

      if (no_motion_counter_ > options_.backend_zero_motion_check_n_frames && !box_test_prior)
      {
        image_motion_detector_stationary_ = true;
        VLOG(5) << "Image is not moving: adding zero velocity prior.";
        if (!backend_.addVelocityPrior(
                createNFrameId(frame_bundle->getBundleId()),
                Eigen::Matrix<FloatType, 3, 1>::Zero(), sigma))
        {
          LOG(ERROR) << "Failed to add a zero velocity prior!";
          DEBUG_CHECK(false) << "Not able to add velocity prior";
        }
        else
        {
          velocity_prior_added = true;
        }
      }
    }
    else
    {
      image_motion_detector_stationary_ = false;
      no_motion_counter_ = 0;
    }
  }

  // only use imu-based motion detection when the images are not good
  if (!box_test_prior && !image_motion_detector_stationary_ && imu_motion_detector_stationary_)
  {
    VLOG(5) << "IMU determined stationary, adding prior at time "
            << frame_bundle->at(0)->getTimestampSec() << std::endl;
    if (!backend_.addVelocityPrior(createNFrameId(frame_bundle->getBundleId()),
                                   Eigen::Matrix<FloatType, 3, 1>::Zero(),
                                   0.005))
    {
      LOG(ERROR) << "Failed to add a zero velocity prior!";
      DEBUG_CHECK(false) << "Not able to add velocity prior";
    }
    else
    {
      velocity_prior_added = true;
    }
  }

  if (options_.trace_motion_detector) {
    const MotionDetector::Diagnostics d = motion_detector_
        ? motion_detector_->diagnostics() : MotionDetector::Diagnostics();
    LOG(INFO) << "SVO_ZERO_MOTION image=" << std::setprecision(17) << image_stamp
              << " reason=" << (motion_detector_ ? d.reason : "disabled")
              << " correspondences=" << d.correspondences
              << " moving=" << d.moving << " allowed_moving=" << d.allowed_moving
              << " max_displacement_px=" << d.max_displacement_px
              << " stationary_count=" << no_motion_counter_
              << " imu_stationary=" << imu_motion_detector_stationary_
              << " box_test=" << box_test_prior << " prior_added=" << velocity_prior_added;
  }

  // Adding new landmarks to backend -----------------------------------------
  size_t num_new_observations = 0;
  for (FramePtr& frame : *frame_bundle)
  {
    if (frame->isKeyframe())
    {
      backend_.setKeyframe(createNFrameId(frame->bundleId()), true);
      active_keyframes_.push_back(frame);
      addLandmarksAndObservationsToBackend(frame);
    }
    else
    {
      // add observations for landmarks that are still visible
      for (size_t kp_idx = 0; kp_idx < frame->numFeatures(); ++kp_idx)
      {
        if (frame->landmark_vec_[kp_idx] &&
            backend_.isPointInEstimator(frame->landmark_vec_[kp_idx]->id()))
        {
          if (backend_.addObservation(frame, kp_idx))
          {
            ++num_new_observations;
          }
        }
      }
    }
  }
  VLOG(10) << "Backend: Added " << num_new_observations
           << " continued observation in non-KF to backend.";

  if (options_.skip_optimization_when_tracking_bad)
  {
    if (frame_bundle->numLandmarksInBA() < options_.min_added_measurements)
    {
      LOG(WARNING) << "Too few visual measurements, skip optimization once.";
      skip_optimization_once_ = true;
    }
  }

  if (velocity_prior_added)
  {
    LOG(WARNING) << "Velocity prior added, not skipping optimization.";
    skip_optimization_once_ = false;
  }

  if (global_landmark_value_version_ < Point::global_map_value_version_)
  {
    backend_.updateFixedLandmarks();
    VLOG(1) << "Update fixed landmarks in Ceres backend: "
            << global_landmark_value_version_ << " ==> "
            << Point::global_map_value_version_ << std::endl;
    global_landmark_value_version_ = Point::global_map_value_version_;
  }

  last_added_nframe_images_ = frame_bundle->getBundleId();
  last_added_frame_stamp_ns_ = frame_bundle->getMinTimestampNanoseconds();
  if (options_.trace_handoff && frame_bundle->isKeyframe() &&
      handoff_trace_frames_ < options_.trace_handoff_frames)
    traceHandoff("after_visual_enqueue", frame_bundle, FrameBundlePtr(), Map::Ptr());
  if (g_permon_backend_)
  {
    g_permon_backend_->log("pre_optim_time", timer.stop());
  }
  if (thread_)
  {
    wait_condition_.notify_one();
  }
  else if (backend_.numFrames() > 1 && !active_keyframes_.empty())
  {
    optimizationStep();
  }
}

// Add all landmarks and observations of frame (under certain criteria)
void CeresBackendInterface::addLandmarksAndObservationsToBackend(
    const FramePtr& frame)
{
  // Statistics.
  size_t n_skipped_points_parallax = 0;
  size_t n_skipped_few_obs = 0;
  size_t n_features_already_in_backend = 0;
  size_t n_new_observations = 0;
  size_t n_new_landmarks = 0;
  size_t n_skipped_not_corner = 0;

  std::vector<std::pair<size_t, size_t>> kp_idx_to_n_obs_map_fixed_lm;

  // iterate through all features
  for (size_t kp_idx = 0; kp_idx < frame->numFeatures(); ++kp_idx)
  {
    const PointPtr& point = frame->landmark_vec_[kp_idx];
    const FeatureType& type = frame->type_vec_[kp_idx];

    // check if feature is associated to landmark
    if (point == nullptr)
    {
      continue;
    }

    // check if landmark was already in to backend, if yes just add observation.
    if (backend_.isPointInEstimator(point->id()))
    {
      ++n_features_already_in_backend;
      if (!backend_.addObservation(frame, kp_idx))
      {
        LOG(WARNING) << "Failed to add an observation!";
        continue;
      }
      ++n_new_observations;
    }
    else
    {
      if (isMapPoint(frame->type_vec_[kp_idx]))
      {
        continue;
      }
      if (options_.only_use_corners)
      {
        if (frame->type_vec_[kp_idx] != FeatureType::kCorner ||
            frame->type_vec_[kp_idx] != FeatureType::kFixedLandmark)
        {
          ++n_skipped_not_corner;
          continue;
        }
      }

      // check if we have enough observations. Might not be the case if seed
      // original frame was already dropped.
      if (point->obs_.size() < options_.min_num_obs)
      {
        VLOG(10) << "Point with less than " << options_.min_num_obs
                 << " observations! Only have " << point->obs_.size();
        ++n_skipped_few_obs;
        continue;
      }

      DEBUG_CHECK(!std::isnan(point->pos_[0])) << "Point is nan!";

      //      //! @todo tune this parameter, do we need it?
      //      if(point->getTriangulationParallax() <
      //      options_.min_parallax_thresh)
      //      {
      //        ++n_skipped_points_parallax;
      //        continue;
      //      }

      //! @todo We should first get all candidate points and sort them
      //!   according to parallax angle and num observations. afterwards only
      //!   add best N observations.
      // add the landmark
      if (isFixedLandmark(type))
      {
        kp_idx_to_n_obs_map_fixed_lm.emplace_back(
            std::make_pair(kp_idx, point->obs_.size()));
        continue;
      }
      if (!backend_.addLandmark(point, false))
      {
        LOG(ERROR) << "Failed to add a landmark!";
        continue;
      }
      ++n_new_landmarks;
      // add an observation to the landmark
      if (!backend_.addObservation(frame, kp_idx))
      {
        LOG(ERROR) << "Failed to add an observation!";
        continue;
      }
      ++n_new_observations;
    }
  }  // landmarks

  // for fixed landmarks
  std::sort(kp_idx_to_n_obs_map_fixed_lm.begin(),
            kp_idx_to_n_obs_map_fixed_lm.end(),
            [](const std::pair<size_t, size_t>& p1,
               const std::pair<size_t, size_t>& p2) {
              return p1.second > p2.second;
            });

  size_t n_added_fixed_lm = 0;
  for (size_t idx = 0; idx < kp_idx_to_n_obs_map_fixed_lm.size(); idx++)
  {
    const size_t cur_kp_idx = kp_idx_to_n_obs_map_fixed_lm[idx].first;
    backend_.addLandmark(frame->landmark_vec_[cur_kp_idx], true);
    backend_.addObservation(frame, cur_kp_idx);
    n_added_fixed_lm++;
    if (backend_.numFixedLandmarks() >=
        optimizer_options_.max_fixed_lm_in_ceres_)
    {
      break;
    }
  }

  VLOG(6) << "Backend has: " << backend_.numFixedLandmarks()
          << " fixed landmarks out of " << backend_.numLandmarks() << std::endl;
  VLOG(6) << "Backend: Added " << n_new_landmarks << " new landmarks";
  VLOG(6) << "Backend: Added " << n_new_observations << " new observations";
  VLOG(6) << "Backend: Observations already in backend: "
          << n_features_already_in_backend;
  VLOG(6) << "Backend: Adding points. Skipped because less than "
          << options_.min_num_obs << " observations: " << n_skipped_few_obs;
  VLOG(6) << "Backend: Adding points. Skipped because small parallax: "
          << n_skipped_points_parallax;
  VLOG(6) << "Backend: Adding points. Skipped because not corner: "
          << n_skipped_not_corner;
}

// Introduce a state for the frame_bundle in backend. Add IMU terms.
void CeresBackendInterface::setInitialBiases(
    const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel)
{
  std::lock_guard<std::mutex> lock(mutex_backend_);
  CHECK_EQ(backend_.numFrames(), 0u);
  backend_.setInitialGyroscopeBias(gyro);
  backend_.setInitialAccelerometerBias(accel);
  imu_handler_->setGyroscopeBias(gyro);
  imu_handler_->setAccelerometerBias(accel);
  LOG(INFO) << "SVO_INIT_BIAS source=frozen_stationary_calibration gyro=["
            << gyro.transpose() << "] accel=[" << accel.transpose() << "]";
}

void CeresBackendInterface::setInitialGravityMeasurements(
    const ImuMeasurements& measurements, bool initialize_gyro_bias,
    bool initialize_radial_accel_bias)
{
  std::lock_guard<std::mutex> lock(mutex_backend_);
  CHECK_EQ(backend_.numFrames(), 0u);
  CHECK_GE(measurements.size(), 2u);
  initial_gravity_measurements_ = measurements;
  if (initialize_radial_accel_bias)
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& m : measurements) mean += m.linear_acceleration_;
    mean /= measurements.size();
    CHECK(mean.allFinite());
    CHECK_GT(mean.norm(), 1.0);
    // A single static orientation identifies only this assumed radial residual.
    // This is not a full three-axis accelerometer calibration.
    const Eigen::Vector3d bias = mean -
        imu_handler_->imu_calib_.gravity_magnitude * mean.normalized();
    backend_.setInitialAccelerometerBias(bias);
    imu_handler_->setAccelerometerBias(bias);
    LOG(INFO) << std::setprecision(17)
              << "SVO_INIT_ACCEL_BIAS source=stationary_radial_residual bias_m_s2=["
              << bias.transpose() << "] mean_norm=" << mean.norm()
              << " gravity=" << imu_handler_->imu_calib_.gravity_magnitude;
  }
  if (initialize_gyro_bias)
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& m : measurements) mean += m.angular_velocity_;
    mean /= measurements.size();
    CHECK(mean.allFinite());
    backend_.setInitialGyroscopeBias(mean);
    imu_handler_->setGyroscopeBias(mean);
    LOG(INFO) << std::setprecision(17)
              << "SVO_INIT_GYRO_BIAS source=stationary_gravity_window mean_rad_s=["
              << mean.transpose() << "] count=" << measurements.size();
  }
}

bool CeresBackendInterface::addStatesAndInertialMeasurementsToBackend(
    const FrameBundlePtr& frame_bundle)
{
  // Gather required IMU measurements ----------------------------------------
  ImuMeasurements imu_measurements;
  const double current_frame_bundle_stamp =
      frame_bundle->getMinTimestampSeconds();

  if (!imu_handler_->waitTill(current_frame_bundle_stamp))
  {
    return false;
  }

  // Get measurements, newest is interpolated to exactly match timestamp of
  // frame_bundle
  if (!imu_handler_->getMeasurementsContainingEdges(current_frame_bundle_stamp,
                                                    imu_measurements, true))
  {
    LOG(ERROR) << "Could not retrieve IMU measurements."
               << " Last frame was at " << last_added_frame_stamp_ns_
               << ", current is at "
               << frame_bundle->getMinTimestampNanoseconds();
    return false;
  }

  // introduce a state for the frame in the backend --------------------------
  // addStates initializes gravity (without propagation) for its first state.
  // Avoid averaging pre-START history for that one operation. Still retrieve
  // the ordinary edge-bracketed measurements above to preserve subsequent
  // propagation endpoints and normal buffer bookkeeping.
  const bool fresh_gravity = backend_.numFrames() == 0 &&
      !initial_gravity_measurements_.empty();
  const ImuMeasurements& state_measurements = fresh_gravity
      ? initial_gravity_measurements_ : imu_measurements;
  if (!backend_.addStates(frame_bundle, state_measurements,
                          current_frame_bundle_stamp))
  {
    LOG(ERROR) << "Failed to add state. Will drop frames.";
    return false;
  }

  if (fresh_gravity)
  {
    LOG(INFO) << "SVO_INIT_GRAVITY phase=backend_consumed count="
              << initial_gravity_measurements_.size()
              << " first_imu=" << std::setprecision(17)
              << initial_gravity_measurements_.back().timestamp_
              << " last_imu=" << initial_gravity_measurements_.front().timestamp_
              << " image=" << current_frame_bundle_stamp;
    initial_gravity_measurements_.clear();
  }

  VLOG(10) << "Backend: Added " << imu_measurements.size() << " inertial "
                                                              "measurements.";
  return true;
}

void CeresBackendInterface::updateFrameStateWithBackend(
    const FramePtr& f, const bool get_speed_bias)
{
  Transformation T_WS;
  bool success = backend_.get_T_WS(f->bundleId(), T_WS);
  T_WS.getRotation().normalize();
  DEBUG_CHECK(success) << "Could not get state for frame bundle "
                       << f->bundleId() << " from backend";
  f->set_T_w_imu(T_WS);
  if (get_speed_bias)
  {
    SpeedAndBias speed_bias;
    success = backend_.getSpeedAndBias(f->bundleId(), speed_bias);
    DEBUG_CHECK(success) << "Could not get speed/bias for frame bundle "
                         << f->bundleId() << " from backend";
    const Eigen::Vector3d velocity_world =
        options_.rotate_world_velocity_for_frame_state
            ? T_WS.getRotation().rotate(speed_bias.block<3, 1>(0, 0))
            : speed_bias.block<3, 1>(0, 0);
    f->setIMUState(velocity_world,
                   speed_bias.block<3, 1>(3, 0), speed_bias.block<3, 1>(6, 0));
  }
}

void CeresBackendInterface::updateBundleStateWithBackend(
    const FrameBundlePtr& frames, const bool get_speed_bias)
{
  Transformation T_WS;
  bool success = backend_.get_T_WS(frames->getBundleId(), T_WS);
  DEBUG_CHECK(success) << "Could not get state for frame bundle "
                       << frames->getBundleId() << " from backend";
  frames->set_T_W_B(T_WS);

  if (get_speed_bias)
  {
    SpeedAndBias speed_bias;
    success = backend_.getSpeedAndBias(frames->getBundleId(), speed_bias);
    DEBUG_CHECK(success) << "Could not get speed/bias for frame bundle "
                         << frames->getBundleId() << " from backend";
    const Eigen::Vector3d velocity_world =
        options_.rotate_world_velocity_for_frame_state
            ? T_WS.getRotation().rotate(speed_bias.block<3, 1>(0, 0))
            : speed_bias.block<3, 1>(0, 0);
    frames->setIMUState(velocity_world,
                        speed_bias.block<3, 1>(3, 0),
                        speed_bias.block<3, 1>(6, 0));
  }
}
void CeresBackendInterface::reset()
{
  VLOG(1) << "Backend: Reset";
  //! @todo implement!
  LOG(ERROR) << "Resetting ceres backend not implemented";
}

void CeresBackendInterface::startThread()
{
  CHECK(thread_ == nullptr) << "Tried to start thread that is already running!";
  stop_thread_ = false;
  thread_.reset(
      new std::thread(&CeresBackendInterface::optimizationLoop, this));
}

void CeresBackendInterface::quitThread()
{
  VLOG(1) << "Interrupting and stopping optimization thread.";
  stop_thread_ = true;
  if (thread_ != nullptr)
  {
    wait_condition_.notify_all();
    thread_->join();
    thread_.reset();
  }
  VLOG(1) << "Thread stopped and joined.";
}

// Performance monitor for benchmarking
void CeresBackendInterface::setPerformanceMonitor(const std::string& trace_dir)
{
  // Paused callbacks can revisit this method; resets construct a new backend.
  // Neither case may truncate the previous initialization's evidence.
  if (g_permon_backend_) return;
  // Initialize Performance Monitor
  g_permon_backend_.reset(new vk::PerformanceMonitor());
  g_permon_backend_->addLog("tot_time");
  g_permon_backend_->addLog("ceres_time");
  g_permon_backend_->addLog("pre_optim_time");
  g_permon_backend_->addLog("marginalization");
  g_permon_backend_->addLog("fixation");
  g_permon_backend_->addLog("n_fixed_lm");
  g_permon_backend_->addLog("bundle_id");
  g_permon_backend_->addLog("image_stamp_ns");
  for (const auto k : MarginalizationTiming::names_)
  {
    g_permon_backend_->addLog(k);
  }
  std::string trace_name = "trace_backend";
  if (std::ifstream(trace_dir + "/trace_backend.csv").good())
    trace_name += "_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
  g_permon_backend_->init(trace_name, trace_dir);
}

void CeresBackendInterface::optimizationLoop()
{
  VLOG(1) << "Backend: Optimization thread started.";
  while (!stop_thread_)
  {
    {
      std::unique_lock<std::mutex> lock(mutex_backend_);
      wait_condition_.wait(lock, [&] {
        return ((last_added_nframe_images_ != last_optimized_nframe_.load())) ||
               stop_thread_;
      });
      if (stop_thread_)
      {
        return;
      }
      optimizationStep();
    }  // release backend mutex.
  }

  VLOG(1) << "Optimization thread ended.";
}

void CeresBackendInterface::writeConsistencyRecord(const std::string& record)
{
  // Bound extra eMMC use across backend resets in this launch, not per reset.
  const size_t max_bytes = 16u * 1024u * 1024u;
  if (!consistency_trace_open_attempted_) {
    consistency_trace_open_attempted_ = true;
    const std::string path = options_.consistency_trace_dir + "/backend_consistency.jsonl";
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    if (existing.good()) consistency_trace_bytes_ = static_cast<size_t>(existing.tellg());
    consistency_trace_.open(path, std::ios::out | std::ios::app);
  }
  struct statvfs space;
  const bool disk_ok = statvfs(options_.consistency_trace_dir.c_str(), &space) == 0 &&
      static_cast<uint64_t>(space.f_bavail) * space.f_frsize >= 512ull * 1024ull * 1024ull;
  if (consistency_trace_ && disk_ok &&
      consistency_trace_bytes_ + record.size() + 1 <= max_bytes) {
    consistency_trace_ << record << '\n';
    consistency_trace_.flush();
    consistency_trace_bytes_ += record.size() + 1;
  }
  // The workstation bag is authoritative even after the local file hits its cap.
  if (publisher_) publisher_->publishConsistency(record);
}

void CeresBackendInterface::optimizationStep()
{
  vk::Timer timer;
  const auto diagnostic_start = std::chrono::steady_clock::now();
  bool consistency_sample = options_.trace_consistency &&
      (diagnostic_start - consistency_last_sample_ >= std::chrono::seconds(1));
  std::ostringstream consistency;
  bool first_snapshot = true;
  bool consistency_started = false;
  double diagnostic_ms = 0;
  const auto snapshot = [&](const char* phase, bool residuals) {
    if (!consistency_sample) return;
    const auto snapshot_start = std::chrono::steady_clock::now();
    if (!consistency_started) {
      consistency_started = true;
      consistency << "{\"schema\":1,\"kind\":\"optimization\",\"instance\":\""
                  << consistency_instance_ << "\",\"cycle\":"
                  << ++consistency_sequence_ << ",\"bundle\":" << last_added_nframe_images_
                  << ",\"image_stamp_ns\":\"" << last_added_frame_stamp_ns_
                  << "\",\"snapshots\":[";
    }
    if (!first_snapshot) consistency << ',';
    first_snapshot = false;
    consistency << "{\"phase\":\"" << phase << "\",\"backend\":"
                << backend_.diagnosticSnapshot(24, residuals) << '}';
    diagnostic_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-snapshot_start).count();
  };
  std::string world_correction = "null";
  {
    std::lock_guard<std::mutex> lock(w_T_correction_mut_);
    if (is_w_T_valid_)
    {
      // Explicit world transformations are never missed by the 1 Hz sampling.
      if (options_.trace_consistency) consistency_sample = true;
      snapshot("before_world_transform", false);
      if (consistency_sample) {
        std::ostringstream transform;
        const auto q = w_T_correction_to_apply_.getRotation().toImplementation();
        const auto p = w_T_correction_to_apply_.getPosition();
        transform << std::setprecision(12) << "{\"p\":[" << p.x() << ',' << p.y() << ',' << p.z()
                  << "],\"q_xyzw\":[" << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << "]}";
        world_correction = transform.str();
      }
      backend_.removeAllPoseFixation();
      backend_.transformMap(
          w_T_correction_to_apply_,
          optimizer_options_.remove_marginalization_term_after_correction_,
          optimizer_options_.recalculate_imu_terms_after_loop);
      for (const FramePtr& f : active_keyframes_)
      {
        f->accumulated_w_T_correction_ =
            w_T_correction_to_apply_ * f->accumulated_w_T_correction_;
      }
      backend_.setOldestFrameFixed();
      is_w_T_valid_ = false;
      snapshot("after_world_transform", false);
    }
  }

  snapshot("before_marginalization", false);
  timer.start();
  MarginalizationTiming mag_timing;
  if (optimizer_options_.marginalize)
  {
    if (!backend_.applyMarginalizationStrategy(
            optimizer_options_.num_keyframes,
            optimizer_options_.num_imu_frames + 1, &mag_timing))
    {
      LOG(ERROR) << "Marginalization failed!";
    }
    updateActiveKeyframes();
  }
  snapshot("after_marginalization", false);
  if (g_permon_backend_)
  {
    g_permon_backend_->log("marginalization", timer.stop());
    for (const auto k : MarginalizationTiming::names_)
      g_permon_backend_->log(k, mag_timing.get(k));
  }

  timer.start();
  if (!backend_.needPoseFixation())
  {
    backend_.removeAllPoseFixation();
    backend_.setAllFixedLandmarksEnabled(true);
    lock_to_fixed_landmarks_ = true;
  }
  else
  {
    backend_.setAllFixedLandmarksEnabled(false);
    if (!backend_.hasFixedPose())
      backend_.setOldestFrameFixed();
    lock_to_fixed_landmarks_ = false;
  }
  if (g_permon_backend_)
    g_permon_backend_->log("fixation", timer.stop());

  if (g_permon_backend_)
    g_permon_backend_->log("n_fixed_lm", backend_.numFixedLandmarks());
  snapshot("before_optimization", true);
  const bool optimization_skipped = skip_optimization_once_;
  timer.start();
  if (skip_optimization_once_)
  {
    skip_optimization_once_ = false;
  }
  else
  {
    backend_.optimize(optimizer_options_.num_iterations,
                      optimizer_options_.num_threads,
                      optimizer_options_.verbose);
  }
  if (g_permon_backend_)
    g_permon_backend_->log("ceres_time", timer.stop());
  snapshot("after_optimization", true);

  last_optimized_nframe_.store(last_added_nframe_images_);
  if (g_permon_backend_)
  {
    g_permon_backend_->log("bundle_id", last_optimized_nframe_.load());
    g_permon_backend_->log("image_stamp_ns", last_added_frame_stamp_ns_);
    g_permon_backend_->log(
        "tot_time", timers_[last_optimized_nframe_.load()].stop());
    g_permon_backend_->writeToFile();
    timers_.erase(timers_.find(last_optimized_nframe_.load()));
  }

  Transformation T_WS;
  bool success = backend_.get_T_WS(last_optimized_nframe_.load(), T_WS);
  DEBUG_CHECK(success) << "Could not get latest Transformation from ceres optimizer";
  SpeedAndBias speed_and_bias;
  success = backend_.getSpeedAndBias(last_optimized_nframe_, speed_and_bias);
  DEBUG_CHECK(success) << "Could not get latest speed/bias from ceres optimizer";
  last_state_.set_T_W_B(T_WS);
  last_state_.set_W_v_B(speed_and_bias.head<3>());
  last_state_.setAccBias(speed_and_bias.tail<3>());
  last_state_.setGyroBias(speed_and_bias.segment<3>(3));

  if (publisher_)
  {
    publisher_->publish(last_state_, last_added_frame_stamp_ns_,
                        last_optimized_nframe_.load());
  }
  if (consistency_sample) {
    const auto q = T_WS.getRotation().toImplementation();
    const auto p = T_WS.getPosition();
    consistency << std::setprecision(12) << "],\"world_correction\":" << world_correction
                << ",\"optimization_skipped\":" << (optimization_skipped ? "true" : "false")
                << ",\"published\":{\"bundle\":" << last_optimized_nframe_.load()
                << ",\"image_stamp_ns\":\"" << last_added_frame_stamp_ns_
                << "\",\"p\":[" << p.x() << ',' << p.y() << ',' << p.z()
                << "],\"q_xyzw\":[" << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w()
                << "],\"velocity_gyro_bias_acc_bias\":[";
    for (int k = 0; k < 9; ++k) { if (k) { consistency << ','; } consistency << speed_and_bias[k]; }
    consistency << "]},\"diagnostic_ms\":" << diagnostic_ms << ",\"cycle_wall_ms\":"
                << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-diagnostic_start).count()
                << '}';
    writeConsistencyRecord(consistency.str());
    consistency_last_sample_ = diagnostic_start;
  }
}

// Set the IMU and the parameters in backend
void CeresBackendInterface::setImu(
    const std::shared_ptr<ImuHandler> imu_handler)
{
  imu_handler_ = imu_handler;

  ImuParameters imu_parameters;
  imu_parameters.a_max = imu_handler_->imu_calib_.saturation_accel_max;
  imu_parameters.g_max = imu_handler_->imu_calib_.saturation_omega_max;
  imu_parameters.sigma_g_c = imu_handler_->imu_calib_.gyro_noise_density;
  imu_parameters.sigma_bg = imu_handler_->imu_init_.omega_bias_sigma;
  imu_parameters.sigma_a_c = imu_handler_->imu_calib_.acc_noise_density;
  imu_parameters.sigma_ba = imu_handler_->imu_init_.acc_bias_sigma;
  imu_parameters.sigma_gw_c =
      imu_handler_->imu_calib_.gyro_bias_random_walk_sigma;
  imu_parameters.sigma_aw_c =
      imu_handler_->imu_calib_.acc_bias_random_walk_sigma;
  imu_parameters.g = imu_handler_->imu_calib_.gravity_magnitude;
  imu_parameters.a0 = imu_handler_->getAccelerometerBias();
  imu_parameters.rate = imu_handler_->imu_calib_.imu_rate;
  imu_parameters.delay_imu_cam = imu_handler_->imu_calib_.delay_imu_cam;
  backend_.addImu(imu_parameters);
}

// Start timer for benchmarking
void CeresBackendInterface::startTimer(const BundleId bundle_id)
{
  timers_[bundle_id].start();
}

void CeresBackendInterface::setCorrectionInWorld(
    const Transformation& w_T_correction)
{
  std::lock_guard<std::mutex> lock(w_T_correction_mut_);
  w_T_correction_to_apply_ = w_T_correction;
  is_w_T_valid_ = true;
}

void CeresBackendInterface::getAllActiveKeyframes(
    std::vector<FramePtr>* keyframes)
{
  CHECK_NOTNULL(keyframes);
  keyframes->clear();
  keyframes->insert(keyframes->begin(), active_keyframes_.begin(),
                    active_keyframes_.end());
}

// Keep track of active keyframes (single frames that were frontend keyframes
// and that are still in backend)
void CeresBackendInterface::updateActiveKeyframes()
{
  // Find the oldest keyframe in backend
  BundleId oldest_keyframe_in_backend = backend_.oldestKeyframeBundleId();
  if (oldest_keyframe_in_backend == -1)
  {
    active_keyframes_.clear();
    return;
  }
  // NFrames are only marginalized sequentially, delete until ID matches
  while (!active_keyframes_.empty())
  {
    if (oldest_keyframe_in_backend == active_keyframes_.front()->bundleId())
    {
      return;
    }
    VLOG(40) << "Backend: marginalized frame with id "
             << active_keyframes_.front()->id();
    active_keyframes_.pop_front();
  }
}

std::string CeresBackendInterface::getStationaryStatusStr() const
{
  std::string str("Static: ");
  if (image_motion_detector_stationary_)
  {
    str += "Visual ";
  }
  if (imu_motion_detector_stationary_)
  {
    str += "Inertial ";
  }

  if (!image_motion_detector_stationary_ && !imu_motion_detector_stationary_)
  {
    str = "Moving";
  }
  return str;
}

void CeresBackendInterface::getLatestSpeedBiasPose(
    Eigen::Matrix<double, 9, 1>* speed_bias, Transformation* T_WS,
    double* timestamp) const
{
  backend_.get_T_WS(backend_.currentBundleId(), *T_WS);
  backend_.getSpeedAndBias(backend_.currentBundleId(), *speed_bias);
  *timestamp = backend_.timestamp(backend_.currentFrameId());
}

void CeresBackendInterface::setReinitStartValues(
    const Eigen::Matrix<double, 9, 1>& sb, const Transformation& Tws,
    const double timestamp)
{
  backend_.is_reinit_ = true;
  backend_.reinit_speed_bias_ = sb;
  backend_.reinit_T_WS_ = Tws;
  backend_.reinit_timestamp_start_ = timestamp;
}

}  // namespace svo
