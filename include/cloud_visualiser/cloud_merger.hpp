#ifndef CLOUD_MERGER_HPP
#define CLOUD_MERGER_HPP

#include <numeric>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/surface/mls.h>

namespace cloud_viz {
  // Merge multiple point clouds together, improving the result at each iteration.
  class CloudMerger
  {
  public:
    using PointT = pcl::PointXYZRGB;
    using PointTNormal = pcl::PointXYZRGBNormal;
    using NormalPt = pcl::Normal;
    using Descriptor = pcl::SHOT1344;

    // ICP error per point pair. Below this error, only ICP will be performed.
    static constexpr double base_error{1.0e-7};

    // ICP parameters.
    static constexpr double icp_tf_epsilon{1e-12};
    static constexpr int icp_max_iter_high{25};
    static constexpr int icp_max_iter_low{40};
    static constexpr double icp_RANSAC_th{0.001};
    static constexpr double icp_max_corr_dist_high{0.2};
    static constexpr double icp_max_corr_dist_low{0.1};
    static constexpr double icp_euclid_fit_epsilon{2.0};

    // TF node.
    struct Node
    {
      size_t id{0}; // ID of the sensor.
      Node* parent{nullptr}; // Parent in the TF tree.
      Eigen::Matrix4f relativePose{Eigen::Matrix4f::Identity()}; // Pose relative to parent.
      Eigen::Matrix4f absolutePose{Eigen::Matrix4f::Identity()}; // Pose relative to world.
      double error{std::numeric_limits<double>::infinity()}; // Error of the current pose.
      size_t numPoints{0}; // Number of cloud points used to compute the last pose.
      bool featureMatching{true}; // True if feature matching needs to be performed.

      pcl::PointCloud<PointTNormal>::Ptr cloudWithNormals;
      pcl::PointCloud<NormalPt>::Ptr normals;
      pcl::PointCloud<PointT>::Ptr keypoints;
      pcl::PointCloud<Descriptor>::Ptr descriptors;
      pcl::CorrespondencesPtr correspondences;
      pcl::CorrespondencesPtr refinedCorrespondences;
    };

    CloudMerger();

    // Merge a new set of clouds.
    void addData(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds);

    const pcl::PointCloud<PointT>::Ptr& result() const;

    // Reset the number of nodes.
    void setNodeSize(size_t size);

  private:
    // Refine a pose estimate.
    void poseRefinement_(const pcl::PointCloud<PointTNormal>::Ptr& sourceCloud,
                         const pcl::PointCloud<PointTNormal>::Ptr& targetCloud,
                         const Eigen::Matrix4f& estimatedPose);

    std::vector<Node> sensorNodes_; // Vector containing all sensors.

    size_t currentIteration_{0}; // Index of the current iteration.
    pcl::PointCloud<PointT>::Ptr result_; // Result from the current iteration.

    // PCL classes.
    pcl::NormalEstimation<PointT, NormalPt> normalEstimator_;
    pcl::ISSKeypoint3D<PointT, PointT, NormalPt> kptDetector_;
    pcl::SHOTColorEstimation<PointT, pcl::Normal, Descriptor> descriptorEstimator_;
    pcl::registration::CorrespondenceEstimation<Descriptor, Descriptor> correspondenceEstimator_;
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> correspondenceRejector_;
    pcl::registration::TransformationEstimationSVD<PointT, PointT> tfEstimator_;
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_;
  }; // namespace cloud_viz
} // namespace cloud_viz

#endif // CLOUD_MERGER_HPP
