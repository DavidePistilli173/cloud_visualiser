#ifndef CLOUD_VISUALISER_CLOUDVISUALISER_HPP
#define CLOUD_VISUALISER_CLOUDVISUALISER_HPP

#include <cloud_visualiser/boxfilter.hpp>
#include <cloud_visualiser/cloud_merger.hpp>
#include <cloud_visualiser/kinect_sensor.hpp>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <utility>

namespace cloud_viz {
  class CloudVisualiser
  {
  public:
    inline static constexpr Box<float> def_filter_box{{-1.0F, 1.0F}, {-1.0F, 1.0F}, {0.5F, 3.5F}};

    CloudVisualiser(const ros::NodeHandle& node_handle);

    bool init();

    void publishFusedCloud();

    void spin();

  private:
    void fuseClouds_(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clouds);

    ros::NodeHandle node_handle_;
    image_transport::ImageTransport image_transport_;

    int num_sensors_;
    std::vector<KinectSensor> sensor_vec_;

    bool fuse_clouds_{false};
    CloudMerger cloud_merger_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_cloud_;
    ros::Publisher cloud_pub_;
  };
} // namespace cloud_viz

#endif
