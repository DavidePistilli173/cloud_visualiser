#include <cloud_visualiser/boxfilter.hpp>
#include <ros/ros.h>

namespace cloud_viz {
  BoxFilter::BoxFilter() { init_(); }

  BoxFilter::BoxFilter(const Box<float>& box)
    : box_(box)
  {
    init_();
  }

  const Box<float>& BoxFilter::box() const { return box_; }
  void BoxFilter::setBox(const Box<float>& box)
  {
    box_ = box;
    filterX_.setFilterLimits(box_.x.first, box_.x.second);
    filterY_.setFilterLimits(box_.y.first, box_.y.second);
    filterZ_.setFilterLimits(box_.z.first, box_.z.second);
  }

  void BoxFilter::filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr secondCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    filterX_.setInputCloud(cloud);
    filterX_.filter(*secondCloud);
    filterY_.setInputCloud(secondCloud);
    filterY_.filter(*cloud);
    filterZ_.setInputCloud(cloud);
    filterZ_.filter(*secondCloud);

    cloud = secondCloud;
  }

  void BoxFilter::init_()
  {
    filterX_.setFilterLimits(box_.x.first, box_.x.second);
    filterX_.setFilterFieldName("x");
    filterY_.setFilterLimits(box_.y.first, box_.y.second);
    filterY_.setFilterFieldName("y");
    filterZ_.setFilterLimits(box_.z.first, box_.z.second);
    filterZ_.setFilterFieldName("z");
  }
} // namespace cloud_viz
