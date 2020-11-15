#include <cloud_visualiser/cloud_viz.hpp>
#include <ros/ros.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cloud_visualiser");
  ros::NodeHandle node_handle("~");

  cloud_viz::CloudVisualiser cv(node_handle);
  if (!cv.init())
    return 1;

  cv.spin();

  return 0;
}
