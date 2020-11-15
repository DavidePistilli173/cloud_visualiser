#ifndef BOXFILTER_HPP
#define BOXFILTER_HPP

#include <iostream>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <utility>

namespace cloud_viz {
  // 3D box.
  template <typename T>
  struct Box
  {
    std::pair<T, T> x;
    std::pair<T, T> y;
    std::pair<T, T> z;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const Box<T>& box)
  {
    os << "x: [" << box.x.first << "," << box.x.second << "]\n";
    os << "y: [" << box.y.first << "," << box.y.second << "]\n";
    os << "z: [" << box.z.first << "," << box.z.second << "]\n";
    return os;
  }

  // 3D passthrough filter.
  class BoxFilter
  {
  public:
    BoxFilter();

    explicit BoxFilter(const Box<float>& box);

    const Box<float>& box() const; // Get the current box.
    void setBox(const Box<float>& box); // Set a new box.

    void filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud); // Filter a cloud.

  private:
    void init_(); // Initialise the filter.

    Box<float> box_{{0.0F, 0.0F}, {0.0F, 0.0F}, {0.0F, 0.0F}}; // Current filtering box.
    pcl::PassThrough<pcl::PointXYZRGB> filterX_; // X axis filter.
    pcl::PassThrough<pcl::PointXYZRGB> filterY_; // Y axis filter.
    pcl::PassThrough<pcl::PointXYZRGB> filterZ_; // Z axis filter.
  };
} // namespace cloud_viz

#endif // BOXFILTER_HPP
