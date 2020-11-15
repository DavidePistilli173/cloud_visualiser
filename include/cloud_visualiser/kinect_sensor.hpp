#ifndef KINECT_SENSOR_HPP
#define KINECT_SENSOR_HPP

#include <calibration_common/pinhole/sensor.h>
#include <cloud_visualiser/boxfilter.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <string>
#include <utility>

namespace cloud_viz {
  class KinectSensor
  {
  public:
    using Ptr = boost::shared_ptr<KinectSensor>;
    using ConstPtr = boost::shared_ptr<const KinectSensor>;

    inline static constexpr float def_leaf_size{0.01F}; // Default leaf size.

    KinectSensor();

    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud() const;

    const Box<float>& filterBox() const; // Get the filtering box.
    void setFilterBox(const Box<float>& box); // Set the filtering box.

    float leafSize() const; // Get the current leaf size.
    void setLeafSize(float leafSize); // Set the leaf size.

    const std::string& name() const; // Get the sensor's name.
    void setName(ros::NodeHandle& nh, const std::string& name); // Set the sensor's name.

    // Set the topic for the RGB image.
    void setRGBSub(image_transport::ImageTransport& image_transport_nh, const std::string& topic);
    // Set the topic for the depth image.
    void setDepthSub(image_transport::ImageTransport& image_transport_nh, const std::string& topic);

    // Receive an RGB image.
    void rgbCallback(const sensor_msgs::Image::ConstPtr& image_msg,
                     const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg);
    // Receive a depth image.
    void depthCallback(const sensor_msgs::Image::ConstPtr& depth_msg);

    bool hasData() const; // Check whether there is enough data to create a cloud.
    void reset(); // Reset the received data.

    void publish(); // Create and publish the most recent point cloud.

  private:
    void init_(); // Initialise the sensor.

    std::string name_; // Name of the sensor.
    image_transport::CameraSubscriber rgb_sub_; // Subscriber to the RGB images.
    image_transport::Subscriber depth_sub_; // Subscriber to the depth images.
    ros::Publisher cloud_pub_; // Publisher of the final clouds.

    calibration::PinholeSensor colour_sensor_; // Colour sensor of the Kinect.
    bool is_sensor_set_{false}; // True if the colour sensor is set.

    cv::Mat rgb_data_; // Last RGB image.
    cv::Mat depth_data_; // Last depth image.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr last_cloud_; // Last computed point cloud.
    bool has_data_{false}; // True if there are both an RGB and a depth image.
    bool has_rgb_data_{false}; // True if there is an RGB image.
    bool has_depth_data_{false}; // True if there is a depth image.

    size_t pub_num_{0}; // Number of published clouds.

    BoxFilter box_filter_;
    float leaf_size_{def_leaf_size}; // Current leaf size.
    pcl::VoxelGrid<pcl::PointXYZRGB> downsampler_;
  };
} // namespace cloud_viz

#endif // KINECT_SENSOR_HPP
