#include <cloud_visualiser/kinect_sensor.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

namespace cloud_viz {
  KinectSensor::KinectSensor() { init_(); }

  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& KinectSensor::cloud() const { return last_cloud_; }

  const Box<float>& KinectSensor::filterBox() const { return box_filter_.box(); }
  void KinectSensor::setFilterBox(const Box<float>& box) { box_filter_.setBox(box); }

  float KinectSensor::leafSize() const { return leaf_size_; }
  void KinectSensor::setLeafSize(float leafSize)
  {
    leaf_size_ = leafSize;
    downsampler_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  }

  const std::string& KinectSensor::name() const { return name_; }
  void KinectSensor::setName(ros::NodeHandle& nh, const std::string& name)
  {
    name_ = name;
    std::stringstream ss;
    ss << name_ << "/cloud";
    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(ss.str(), 1);
  }

  void KinectSensor::setRGBSub(image_transport::ImageTransport& image_transport_nh, const std::string& topic)
  {
    rgb_sub_ = image_transport_nh.subscribeCamera(topic, 1, &KinectSensor::rgbCallback, this);
  }
  void KinectSensor::setDepthSub(image_transport::ImageTransport& image_transport_nh, const std::string& topic)
  {
    depth_sub_ = image_transport_nh.subscribe(topic, 1, &KinectSensor::depthCallback, this);
  }

  void KinectSensor::rgbCallback(const sensor_msgs::Image::ConstPtr& image_msg,
                                 const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg)
  {
    cv_bridge::CvImage::Ptr image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    rgb_data_ = image_ptr->image;

    if (!is_sensor_set_) {
      calibration::PinholeCameraModel::ConstPtr cm =
        boost::make_shared<calibration::PinholeCameraModel>(*camera_info_msg);
      colour_sensor_.setCameraModel(cm);
    }

    has_rgb_data_ = true;
    has_data_ = has_rgb_data_ && has_depth_data_;
  }

  void KinectSensor::depthCallback(const sensor_msgs::Image::ConstPtr& depth_msg)
  {
    cv_bridge::CvImage::Ptr depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_data_ = depth_ptr->image;
    has_depth_data_ = true;
    has_data_ = has_rgb_data_ && has_depth_data_;
  }

  bool KinectSensor::hasData() const { return has_data_; }

  void KinectSensor::reset()
  {
    rgb_data_ = cv::Mat();
    depth_data_ = cv::Mat();
    has_data_ = false;
    has_rgb_data_ = false;
    has_depth_data_ = false;
  }

  void KinectSensor::publish()
  {
    if (rgb_data_.empty() && depth_data_.empty() && !(rgb_data_.rows == depth_data_.rows)
        && !(rgb_data_.cols == depth_data_.cols))
      return;

    // Fuse color and depth images into a point cloud.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->reserve(rgb_data_.rows * rgb_data_.cols);
    for (int y = 0; y < rgb_data_.rows; ++y) {
      for (int x = 0; x < rgb_data_.cols; ++x) {
        pcl::PointXYZRGB point(
          rgb_data_.at<cv::Vec3b>(y, x)[2], rgb_data_.at<cv::Vec3b>(y, x)[1], rgb_data_.at<cv::Vec3b>(y, x)[0]);
        auto world_point =
          colour_sensor_.cameraModel()->projectPixelTo3dRay(calibration::PinholeCameraModel::Point2(x, y));

        point.z = depth_data_.at<unsigned short>(y, x) / 1000.0F;
        point.x = world_point.x() * point.z;
        point.y = world_point.y() * point.z;
        cloud->push_back(point);
      }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (leaf_size_ > 0.0) {
      downsampler_.setInputCloud(cloud);
      downsampler_.filter(*filteredCloud);
    }
    else
      *filteredCloud = *cloud;

    box_filter_.filter(filteredCloud);

    // Remove NaN values.
    std::vector<int> idxs;
    pcl::removeNaNFromPointCloud(*filteredCloud, *filteredCloud, idxs);

    // Save the cloud.
    last_cloud_ = filteredCloud;

    sensor_msgs::PointCloud2 outputMsg;
    pcl::toROSMsg(*filteredCloud, outputMsg);
    outputMsg.header.frame_id = name_;
    outputMsg.header.stamp = ros::Time::now();
    cloud_pub_.publish(outputMsg);
    reset();

    // ROS_INFO("Published %lu points", filteredCloud->points.size());
    // ++pub_num_;
    // ROS_INFO_STREAM("Published cloud n." << pub_num_ << " for sensor " << name_);
  }

  void KinectSensor::init_() { downsampler_.setLeafSize(leaf_size_, leaf_size_, leaf_size_); }
} // namespace cloud_viz
