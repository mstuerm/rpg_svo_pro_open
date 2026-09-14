#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>

#include <ros/ros.h>
#include <std_msgs/String.h>    // user-input
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>


#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>

namespace svo {

// forward declarations
class FrameHandlerBase;
class Visualizer;
class ImuHandler;
class BackendInterface;
class CeresBackendInterface;
class CeresBackendPublisher;

enum class PipelineType {
  kMono,
  kStereo,
  kArray
};

/// SVO Interface
class SvoInterface
{
public:

  // ROS subscription and publishing.
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  PipelineType pipeline_type_;
  ros::Subscriber sub_remote_key_;
  ros::Publisher pub_pampc_landmarks_;
  std::string remote_input_;
  std::unique_ptr<std::thread> imu_thread_;
  std::unique_ptr<std::thread> image_thread_;

  // SVO modules.
  std::shared_ptr<FrameHandlerBase> svo_;
  std::shared_ptr<Visualizer> visualizer_;
  std::shared_ptr<ImuHandler> imu_handler_;
  std::shared_ptr<BackendInterface> backend_interface_;
  std::shared_ptr<CeresBackendInterface> ceres_backend_interface_;
  std::shared_ptr<CeresBackendPublisher> ceres_backend_publisher_;

  CameraBundlePtr ncam_;

  // Parameters
  bool set_initial_attitude_from_gravity_ = true;

  // System state.
  bool quit_ = false;
  bool idle_ = false;
  bool automatic_reinitialization_ = false;
  // Remote reset/start commands arrive on ROS's callback thread while images
  // are processed on image_thread_. Resetting the Ceres backend concurrently
  // with addFrameBundle() invalidates backend state underneath the optimizer.
  std::mutex pipeline_mutex_;
  // Consecutive current-frame observations used by the strict PAMPC
  // landmark filter. This mirrors LeapCSVOEnv::_landmark_persistence.
  std::unordered_map<int, int> pampc_landmark_persistence_;

  SvoInterface(const PipelineType& pipeline_type,
          const ros::NodeHandle& nh,
          const ros::NodeHandle& private_nh);

  virtual ~SvoInterface();

  // Processing
  void processImageBundle(
      const std::vector<cv::Mat>& images,
      int64_t timestamp_nanoseconds);

  bool setImuPrior(const int64_t timestamp_nanoseconds);

  void publishResults(
      const std::vector<cv::Mat>& images,
      const int64_t timestamp_nanoseconds);

  void publishPampcLandmarks(const int64_t timestamp_nanoseconds);

  // Subscription and callbacks
  void monoCallback(const sensor_msgs::ImageConstPtr& msg);
  void monoDepthCallback(const sensor_msgs::ImageConstPtr& image_msg,
                         const sensor_msgs::ImageConstPtr& depth_msg);
  void stereoCallback(
      const sensor_msgs::ImageConstPtr& msg0,
      const sensor_msgs::ImageConstPtr& msg1);
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
  void inputKeyCallback(const std_msgs::StringConstPtr& key_input);


  // These functions are called before and after monoCallback or stereoCallback.
  // a derived class can implement some additional logic here.
  virtual void imageCallbackPreprocessing(int64_t timestamp_nanoseconds) {}
  virtual void imageCallbackPostprocessing() {}

  void subscribeImu();
  void subscribeImage();
  void subscribeRemoteKey();

  void imuLoop();
  void monoLoop();
  void stereoLoop();
};

} // namespace svo
