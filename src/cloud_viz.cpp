#include <cloud_visualiser/cloud_viz.hpp>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <sstream>

namespace cloud_viz {
  CloudVisualiser::CloudVisualiser(const ros::NodeHandle& node_handle)
    : node_handle_(node_handle)
    , image_transport_(node_handle)
  {}

  bool CloudVisualiser::init()
  {
    node_handle_.param("num_sensors", num_sensors_, 0);
    ROS_INFO("Listening to %d sensors.", num_sensors_);
    std::string defParam;
    sensor_vec_.resize(num_sensors_);

    Box<float> filteringBox;
    node_handle_.param("min_x", filteringBox.x.first, def_filter_box.x.first);
    node_handle_.param("max_x", filteringBox.x.second, def_filter_box.x.second);
    node_handle_.param("min_y", filteringBox.y.first, def_filter_box.y.first);
    node_handle_.param("max_y", filteringBox.y.second, def_filter_box.y.second);
    node_handle_.param("min_z", filteringBox.z.first, def_filter_box.z.first);
    node_handle_.param("max_z", filteringBox.z.second, def_filter_box.z.second);
    ROS_INFO_STREAM("Filtering box:\n" << filteringBox);

    float leafSize;
    node_handle_.param("leaf_size", leafSize, KinectSensor::def_leaf_size);

    node_handle_.param("fuse_clouds", fuse_clouds_, false);
    // Activate point cloud fusion only if there are 2 or more sensors.
    fuse_clouds_ = fuse_clouds_ && (num_sensors_ >= 2);
    if (fuse_clouds_) {
      cloud_merger_.setNodeSize(num_sensors_);
      cloud_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>("cloud_viz/cloud", 1);

      // Set the parameters for cloud-merging
      CloudMerger::Params params;
      node_handle_.param("normal_radius", params.normalR, CloudMerger::normal_radius);
      node_handle_.param("kpt_border_radius", params.kptBorderR, CloudMerger::kpt_border_radius);
      node_handle_.param("kpt_non_max_radius", params.kptNonMaxR, CloudMerger::kpt_non_max_radius);
      node_handle_.param("kpt_salient_radius", params.kptSalientR, CloudMerger::kpt_salient_radius);
      node_handle_.param("kpt_search_radius", params.kptSearchR, CloudMerger::kpt_search_radius);
      node_handle_.param("desc_lfr_radius", params.descLFR, CloudMerger::desc_lfr_radius);
      node_handle_.param("desc_search_radius", params.descSearchR, CloudMerger::desc_search_radius);
      node_handle_.param("rej_inlier_th", params.rejInlierTh, CloudMerger::rej_inlier_th);
      node_handle_.param("icp_tf_epsilon", params.icpTFEps, CloudMerger::icp_tf_epsilon);
      node_handle_.param("icp_max_iter_high", params.icpMaxIterH, CloudMerger::icp_max_iter_high);
      node_handle_.param("icp_max_iter_low", params.icpMaxIterL, CloudMerger::icp_max_iter_low);
      node_handle_.param("icp_RANSAC_th", params.icpRANSACTh, CloudMerger::icp_RANSAC_th);
      node_handle_.param("icp_max_corr_dist_high", params.icpMaxCorrDistH, CloudMerger::icp_max_corr_dist_high);
      node_handle_.param("icp_max_corr_dist_low", params.icpMaxCorrDistL, CloudMerger::icp_max_corr_dist_low);
      node_handle_.param("icp_euclid_fit_epsilon", params.icpEuclidFitEps, CloudMerger::icp_euclid_fit_epsilon);

      cloud_merger_.setParams(params);
    }

    for (int i = 0; i < num_sensors_; ++i) {
      std::string paramVal;

      std::stringstream nameParam;
      nameParam << "sensor_" << i << "_name";
      if (!node_handle_.param(nameParam.str(), paramVal, defParam)) {
        ROS_FATAL_STREAM("Missing parameter " << nameParam.str());
        return false;
      }
      ROS_INFO("Adding sensor %s.", paramVal.c_str());
      sensor_vec_[i].setName(node_handle_, paramVal);

      std::stringstream rgbParam;
      rgbParam << "sensor_" << i << "_rgb";
      if (!node_handle_.param(rgbParam.str(), paramVal, defParam)) {
        ROS_FATAL_STREAM("Missing parameter " << rgbParam.str());
        return false;
      }
      ROS_INFO("Lintening to %s.", paramVal.c_str());
      sensor_vec_[i].setRGBSub(image_transport_, paramVal);

      std::stringstream depthParam;
      depthParam << "sensor_" << i << "_depth";
      if (!node_handle_.param(depthParam.str(), paramVal, defParam)) {
        ROS_FATAL_STREAM("Missing parameter " << depthParam.str());
        return false;
      }
      ROS_INFO("Listening to %s.", paramVal.c_str());
      sensor_vec_[i].setDepthSub(image_transport_, paramVal);

      sensor_vec_[i].setFilterBox(filteringBox);
      sensor_vec_[i].setLeafSize(leafSize);
    }

    return true;
  }

  void CloudVisualiser::publishFusedCloud()
  {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*fused_cloud_, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "cloud_viz";

    cloud_pub_.publish(msg);
  }

  void CloudVisualiser::spin()
  {
    ros::Rate rate(1.0);

    while (ros::ok()) {
      ros::spinOnce();

      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
      clouds.reserve(sensor_vec_.size());
      for (auto& sensor : sensor_vec_) {
        if (sensor.hasData()) {
          sensor.publish();
          clouds.emplace_back(sensor.cloud());
        }
      }

      if (fuse_clouds_ && clouds.size() == sensor_vec_.size()) {
        fuseClouds_(clouds);
        publishFusedCloud();
      }
    }
  }

  void CloudVisualiser::fuseClouds_(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clouds)
  {
    ROS_INFO("Started cloud fusion.");
    cloud_merger_.addData(clouds);
    fused_cloud_ = cloud_merger_.result();
  }
} // namespace cloud_viz
